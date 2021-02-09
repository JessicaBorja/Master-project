import os, datetime, logging
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import hydra
from omegaconf import OmegaConf
from affordance_model.datasets import VREnvData
from affordance_model.segmentator import Segmentator
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

@hydra.main(config_path="./config", config_name="cfg_affordance")
def train(cfg):
    print("Running configuration: ", cfg)
    logger = logging.getLogger(__name__)
    logger.info("Running configuration: %s", OmegaConf.to_yaml(cfg) )

    #Data split
    train = VREnvData(split = "train", **cfg.dataset)
    val = VREnvData(split = "validation", **cfg.dataset)
    logger.info('train_data {}'.format(train.__len__()))
    logger.info('val_data {}'.format(val.__len__()))
    
    train_loader = DataLoader(train, **cfg.dataloader)
    val_loader = DataLoader(val, **cfg.dataloader)
    logger.info('train minibatches {}'.format(len(train_loader)))
    logger.info('val minibatches {}'.format(len(val_loader)))

    #Initialize model
    checkpoint_callback = ModelCheckpoint(
        monitor ='val_loss',
        dirpath = "trained_models",
        filename = 'affordance-{epoch:02d}-{val_loss:.4f}',
        save_top_k = 3,
        verbose = True
        )

    model_name = cfg.model_name#
    model_name = "{}_{}".format(model_name, datetime.datetime.now().strftime('%d-%m_%I-%M'))
    tb_logger = TensorBoardLogger("tb_logs", name = model_name)

    aff_model = Segmentator(cfg.model_cfg)
    trainer = pl.Trainer(
        callbacks = [checkpoint_callback],
        logger = tb_logger,
        **cfg.trainer)
    trainer.fit(aff_model, train_loader, val_loader)
        
if __name__ == "__main__":
    train()