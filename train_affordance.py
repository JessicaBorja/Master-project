import datetime
import logging
import hydra
from omegaconf import OmegaConf
from affordance_model.segmentator import Segmentator
from affordance_model.utils.utils import get_loaders
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


@hydra.main(config_path="./config", config_name="cfg_affordance")
def train(cfg):
    print("Running configuration: ", cfg)
    logger = logging.getLogger(__name__)
    logger.info("Running configuration: %s", OmegaConf.to_yaml(cfg))

    # Data split
    train_loader, val_loader = \
        get_loaders(logger, cfg.dataset, cfg.dataloader)

    # Initialize model
    checkpoint_loss_callback = ModelCheckpoint(
        monitor='validation/total_loss',
        dirpath="trained_models",
        filename='affordance-epoch={epoch:02d}-val_loss={validation/total_loss:.2f}',
        save_top_k=2,
        verbose=True
        )

    checkpoint_miou_callback = ModelCheckpoint(
        monitor='validation/mIoU',
        dirpath="trained_models",
        filename='affordance-epoch={epoch:02d}-val_miou={validation/mIoU:.2f}',
        save_top_k=2,
        verbose=True
        )

    model_name = cfg.model_name
    model_name = "{}_{}".format(
                        model_name,
                        datetime.datetime.now().strftime('%d-%m_%H-%M'))  # 24hr format

    wandb_logger = WandbLogger(name=model_name, project="affordance_model")
    # tb_logger = TensorBoardLogger("tb_logs", name=model_name)

    aff_model = Segmentator(cfg.model_cfg, cmd_log=logger)
    trainer = pl.Trainer(
        callbacks=[checkpoint_miou_callback, checkpoint_loss_callback],
        logger=wandb_logger,
        **cfg.trainer)
    trainer.fit(aff_model, train_loader, val_loader)


if __name__ == "__main__":
    train()
