import datetime
import logging
import hydra
from omegaconf import OmegaConf
from affordance_model.segmentator import Segmentator
from affordance_model.datasets import get_loaders
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


def print_cfg(cfg):
    print_cfg = OmegaConf.to_container(cfg)
    print_cfg.pop("dataset")
    print_cfg.pop("trainer")
    print_cfg.pop("optim")
    return OmegaConf.create(print_cfg)


@hydra.main(config_path="./config", config_name="cfg_affordance")
def train(cfg):
    print("Running configuration: ", cfg)
    logger = logging.getLogger(__name__)
    logger.info("Running configuration: %s",
                OmegaConf.to_yaml(print_cfg(cfg)))

    # Data split
    train_loader, val_loader = \
        get_loaders(logger, cfg.dataset, cfg.dataloader, cfg.img_size)

    # 24hr format
    model_name = cfg.model_name

    # Initialize model
    checkpoint_loss_callback = ModelCheckpoint(
        monitor='val_total_loss',
        dirpath="trained_models",
        filename='%s_{epoch:02d}-{val_total_loss:.3f}' % model_name,
        save_top_k=2,
        verbose=True,
        mode='min'
        )

    checkpoint_miou_callback = ModelCheckpoint(
        monitor='val_miou',
        dirpath="trained_models",
        filename='%s_{epoch:02d}-{val_miou:.3f}' % model_name,
        save_top_k=2,
        verbose=True,
        mode='max',
        save_last=True,
        )

    model_name = "{}_{}".format(
                        model_name,
                        datetime.datetime.now().strftime('%d-%m_%H-%M'))
    wandb_logger = WandbLogger(name=model_name, project="affordance_model")
    aff_model = Segmentator(cfg.model_cfg, cmd_log=logger)
    trainer = pl.Trainer(
        callbacks=[checkpoint_miou_callback, checkpoint_loss_callback],
        logger=wandb_logger,
        **cfg.trainer)
    trainer.fit(aff_model, train_loader, val_loader)


if __name__ == "__main__":
    train()
