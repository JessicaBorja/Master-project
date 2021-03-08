import torch
from torch import nn
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import numpy as np


class Segmentator(pl.LightningModule):
    def __init__(self, cfg, cmd_log=None):
        super().__init__()
        # https://github.com/qubvel/segmentation_models.pytorch
        self.unet = None
        self.init_model(n_classes=2)
        self.optimizer_cfg = cfg.optimizer
        self.criterion = nn.CrossEntropyLoss(
                            weight=torch.tensor(cfg.loss_weight))
        self.cmd_log = cmd_log
        self._batch_loss = []
        self._batch_miou = []

    def init_model(self, n_classes=2):
        self.unet = smp.Unet(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=3,  # RGB
            classes=n_classes,
            encoder_depth=3,  # Should be equal to number of layers in decoder
            decoder_channels=(128, 64, 32),
            activation='softmax'
        )
        # Fix encoder weights. Only train decoder
        for name, param in self.unet.named_parameters():
            if name.startswith('encoder'):
                param.requires_grad = False

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        self.unet.encoder.eval()
        features = self.unet.encoder(x)
        decoder_output = self.unet.decoder(*features)
        masks = self.unet.segmentation_head(decoder_output)
        return masks

    def log_stats(self, split, max_batch, batch_idx, loss, miou):
        if(batch_idx >= max_batch - 1):
            e_loss = 0 if len(self._batch_loss) == 0 else np.mean(self._batch_loss)
            e_miou = 0 if len(self._batch_miou) == 0 else np.mean(self._batch_miou)
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
        loss = self.criterion(x_hat, masks)
        mIoU = self.compute_mIoU(x_hat, masks)
        info_dict = {"train_loss": loss,
                     "train_mIoU": mIoU}

        self.log_stats("train", self.trainer.num_training_batches,
                       batch_idx, loss, mIoU)
        self.log_dict(info_dict, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, masks = val_batch
        x_hat = self.unet(x)
        loss = self.criterion(x_hat, masks)
        mIoU = self.compute_mIoU(x_hat, masks)
        info_dict = {"val_loss": loss,
                     "val_mIoU": mIoU}

        self.log_stats("validation", sum(self.trainer.num_val_batches),
                       batch_idx, loss, mIoU)
        self.log_dict(info_dict, on_epoch=True, on_step=False)
        return loss

    def compute_mIoU(self, logits, gt, threshold=0.5):
        if logits.shape[1] == 1 or len(logits.shape) == 3:
            if threshold == 0.5:
                pred = logits.round().byte()
            else:
                pred = logits > threshold
        else:
            pred = logits.argmax(dim=1).byte()
        intersection = ((pred == 1) & (gt == 1)).sum().float()
        union = ((pred == 1) | (gt == 1)).sum().float()
        return intersection/(union+1.)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self.optimizer_cfg)
        return optimizer
