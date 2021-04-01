import torch
from torch import nn
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import numpy as np
from affordance_model.utils.losses import \
    compute_mIoU, pixel2spatial, compute_dice_loss


class Segmentator(pl.LightningModule):
    def __init__(self, cfg, cmd_log=None):
        super().__init__()
        # https://github.com/qubvel/segmentation_models.pytorch
        self.unet = None
        self.init_model(n_classes=2)
        self.optimizer_cfg = cfg.optimizer
        if(cfg.add_dice_loss):
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.CrossEntropyLoss(
                weight=torch.tensor(cfg.loss_weight))
        self.cmd_log = cmd_log
        self._batch_loss = []
        self._batch_miou = []
        self._add_dice_loss = cfg.add_dice_loss
        self._dice_weight = cfg.dice_weight

    def init_model(self, n_classes=2):
        self.unet = smp.Unet(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=1,  # Grayscale
            classes=n_classes,
            encoder_depth=3,  # Should be equal to number of layers in decoder
            decoder_channels=(128, 64, 32),
            activation='softmax2d'
        )
        # Fix encoder weights. Only train decoder
        for param in self.unet.encoder.parameters():
            param.requires_grad = False

    def compute_loss(self, preds, labels):
        # Preds = (B, C, W, H)
        loss = self.criterion(preds, labels)
        info = {"CE_loss": loss}
        if self._add_dice_loss:
            # Unweighted cross entropy + dice loss
            label_spatial = pixel2spatial(
                                    labels,
                                    preds.shape[2],
                                    preds.shape[3])
            dice_loss = compute_dice_loss(label_spatial, preds)
            loss += self._dice_weight * dice_loss
            info["dice_loss"] = dice_loss
        return loss, info

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        self.unet.encoder.eval()
        features = self.unet.encoder(x)
        decoder_output = self.unet.decoder(*features)
        masks = self.unet.segmentation_head(decoder_output)
        return masks

    def log_stats(self, split, max_batch, batch_idx, loss, miou):
        if(batch_idx >= max_batch - 1):
            e_loss = 0 if len(self._batch_loss) == 0 \
                else np.mean(self._batch_loss)
            e_miou = 0 if len(self._batch_miou) == 0 \
                else np.mean(self._batch_miou)
            self.cmd_log.info(
                "%s [epoch %4d]" % (split, self.current_epoch) +
                "loss: %.3f, mIou: %.3f" % (e_loss, e_miou))
            self._batch_loss = []
            self._batch_loss = []
        else:
            self._batch_loss.append(loss.item())
            self._batch_miou.append(miou.item())

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x, masks = batch
        x_hat = self.unet(x)  # B, N_classes, img_size, img_size
        total_loss, info = self.compute_loss(x_hat, masks)
        mIoU = compute_mIoU(x_hat, masks)

        self.log_stats("train", self.trainer.num_training_batches,
                       batch_idx, total_loss, mIoU)
        self.log("train/loss", total_loss, on_step=False, on_epoch=True)
        self.log("train/mIoU", mIoU, on_step=False, on_epoch=True)
        for k, v in info.items():
            self.log("train/%s" % k, v, on_step=False, on_epoch=True)

        return total_loss

    def validation_step(self, val_batch, batch_idx):
        x, masks = val_batch
        x_hat = self.unet(x)
        total_loss, info = self.compute_loss(x_hat, masks)
        mIoU = compute_mIoU(x_hat, masks)

        # Log metrics
        self.log_stats("validation", sum(self.trainer.num_val_batches),
                       batch_idx, total_loss, mIoU)
        self.log("validation/mIoU", mIoU, on_step=False, on_epoch=True)
        self.log("validation/total_loss", total_loss,
                 on_step=False, on_epoch=True)
        for k, v in info.items():
            self.log("validation/%s" % k, v, on_step=False, on_epoch=True)

        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self.optimizer_cfg)
        return optimizer
