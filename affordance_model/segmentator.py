import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

class Segmentator(pl.LightningModule):
  def __init__(self, cfg):
    super().__init__()
    #https://github.com/qubvel/segmentation_models.pytorch
    self.unet = None
    self.init_model()    
    self.optimizer_cfg = cfg.optimizer
    self.criterion = nn.CrossEntropyLoss()

  def init_model(self):
    self.unet = smp.Unet(
      encoder_name ="resnet18",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
      encoder_weights ="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
      in_channels = 3,                 # model input channels (1 for gray-scale images, 3 for RGB, etc.)
      classes = 2,                     # model output channels (number of classes in your dataset)
      encoder_depth = 3,               # Should be equal to number of layers in decoder
      decoder_channels = (128, 64, 32),
      activation = 'softmax'
    )
  
    for name, param in self.unet.named_parameters(): #Fix encoder weights. Only train decoder
      if name.startswith('encoder'):
        param.requires_grad = False

  def forward(self, x):
    # in lightning, forward defines the prediction/inference actions
    self.unet.encoder.eval()
    features = self.unet.encoder(x)
    decoder_output = self.unet.decoder(*features)
    masks = self.unet.segmentation_head(decoder_output)
    return masks

  def training_step(self, batch, batch_idx):
    # training_step defined the train loop. It is independent of forward
    x, masks = batch
    x_hat = self.unet(x) # B, N_classes, img_size, img_size
    loss = self.criterion(x_hat, masks)
    mIoU = self.compute_mIoU(x_hat, masks)
    info_dict = {"train_loss": loss,
                 "train_mIoU": mIoU}
    self.log_dict(info_dict, on_epoch = True, on_step = False)
    return loss
  
  def validation_step(self, val_batch, batch_idx):
    x, masks = val_batch
    x_hat = self.unet(x)
    loss = self.criterion(x_hat, masks)
    mIoU = self.compute_mIoU(x_hat, masks)
    info_dict = {"val_loss": loss,
                 "val_mIoU": mIoU}
    self.log_dict(info_dict, on_epoch = True, on_step = False)
    return loss

  def compute_mIoU(self, logits, gt, threshold=0.5):
    if logits.shape[1] == 1 or len(logits.shape)==3:
        if threshold==0.5:
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