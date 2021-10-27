
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from hpbandster.core.worker import Worker
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import BOHB
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import numpy as np
import pickle
import hydra
import os
import logging
import torch
import datetime
from omegaconf import OmegaConf
from vapo.affordance_model.utils.losses import compute_mIoU
from vapo.affordance_model.segmentator_centers import Segmentator
from vapo.affordance_model.datasets import get_loaders
log = logging.getLogger(__name__)


class AffWorker(Worker):
    def __init__(self, cfg, run_id, nameserver, logger):
        super().__init__(run_id, nameserver, logger=logger)
        self.cfg = cfg
        self.logger = logger

    def validate(val_loader, model):
        batch_miou = [], []
        for x, masks in val_loader:
            with torch.no_grad():
                x_hat = model(x)
            mIoU = compute_mIoU(x_hat, masks)
            batch_miou.append(mIoU)
        return np.mean(batch_miou)

    def compute(self, config, budget, working_directory, *args, **kwargs):
        log.info("Running job/config_id: " + str(kwargs["config_id"]))
        log.info(config)

        # Model and hyperpameters
        model_cfg = self.cfg.model_cfg
        model_cfg.aff_loss.dice_loss.weight = config.pop("dice_loss")
        model_cfg.aff_loss.ce_loss.weight = config.pop("ce_loss")
        model_cfg.centers_loss.weight = config.pop("centers_loss")
        model_cfg.optimizer.lr = config.pop("lr")

        # Load model
        model = Segmentator(model_cfg, cmd_log=self.logger)

        # Get loaders
        dataloader_cfg = {"batch_size": config.pop("batch_size"),
                          **self.cfg.optim.dataloader}
        img_size = self.cfg.img_size[self.cfg.dataset.cam]
        train_loader, val_loader = \
            get_loaders(log, self.cfg.dataset,
                        dataloader_cfg,
                        img_size,
                        self.cfg.model_cfg.n_classes)

        # Set callbacks
        checkpoint_loss_callback = ModelCheckpoint(
            monitor='validation/total_loss',
            dirpath="trained_models",
            filename='affordance-{epoch:02d}-{val_loss:.4f}',
            save_top_k=2,
            verbose=True
            )

        checkpoint_miou_callback = ModelCheckpoint(
            monitor='validation/mIoU',
            dirpath="trained_models",
            filename='affordance-{epoch:02d}-{val_miou:.4f}',
            save_top_k=2,
            verbose=True
            )

        model_name = self.cfg.model_name
        # 24hr format
        curr_date = datetime.datetime.now().strftime('%d-%m_%H-%M')
        model_name = "{}_{}".format(model_name, curr_date)

        wandb_logger = WandbLogger(
                            name=model_name,
                            project="affordance_model")
        # Train configuration
        callbacks = [checkpoint_miou_callback, checkpoint_loss_callback]
        trainer_cfg = OmegaConf.to_container(self.cfg.trainer)
        trainer_cfg.pop("max_epochs")
        train_cfg = OmegaConf.create({"max_epochs": budget,
                                      **trainer_cfg})
        trainer = pl.Trainer(
                    callbacks=callbacks,
                    logger=wandb_logger,
                    accelerator="ddp",
                    **train_cfg)

        # Train and validate
        trainer.fit(model, train_loader, val_loader)
        miou = self.validate(self.val_loader, model)

        # remember: HpBandSter always minimizes!
        # want to maximize intersection over union
        return ({'loss': - miou})

    @staticmethod
    def get_configspace():
        """
        It builds the configuration space with the needed hyperparameters.
        :return: ConfigurationsSpace-Object
        """
        cs = CS.ConfigurationSpace()

        batch_size = CSH.UniformIntegerHyperparameter('batch_size',
                                                      lower=32, upper=256)
        lr = CSH.UniformFloatHyperparameter('lr',
                                            lower=1e-6, upper=1e-4,
                                            log=True)

        # Dice Loss
        dice_weight = CSH.UniformIntegerHyperparameter('dice_loss',
                                                       lower=2, upper=7)
        ce_loss = CSH.UniformIntegerHyperparameter('ce_loss',
                                                   lower=1, upper=6)

        # Centers loss
        centers_loss = CSH.UniformIntegerHyperparameter('centers_loss',
                                                        lower=2, upper=6)

        # Conditions
        cs.add_hyperparameters([batch_size, lr, dice_weight,
                                ce_loss, centers_loss])
        return cs


def optimize(trial_name, cfg, run_id, nameserver, max_budget=100, min_budget=5,
             n_iterations=3, n_workers=1, worker=False):

    result_logger = hpres.json_result_logger(directory=".", overwrite=False)
    logging.basicConfig(level=logging.DEBUG)

    # Start a nameserver (see example_1)
    NS = hpns.NameServer(run_id=run_id, host=nameserver, port=None)
    NS.start()

    w = AffWorker(cfg,
                  nameserver=nameserver,
                  run_id=run_id,
                  logger=log)  # , id=i)
    w.run(background=True)

    bohb = BOHB(configspace=w.get_configspace(),
                run_id=run_id,
                nameserver=nameserver,
                result_logger=result_logger,
                min_budget=min_budget,
                max_budget=max_budget)

    res = bohb.run(n_iterations=n_iterations)  # , min_n_workers = n_workers)
    # store results
    if not os.path.exists("./optimization_results/"):
        os.makedirs("./optimization_results/")
    with open(os.path.join("./optimization_results/", "%s.pkl" % trial_name),
              'wb') as fh:
        pickle.dump(res, fh)

    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()
    # id2config = res.get_id2config_mapping()
    incumbent = str(res.get_incumbent_id())
    print("incumbent: ", incumbent)
    log.info("Finished optimization, incumbent: %s" % incumbent)


def read_results(name):
    with open(os.path.join("./optimization_results/", name), 'rb') as fh:
        res = pickle.load(fh)

    id2config = res.get_id2config_mapping()
    incumbent = res.get_incumbent_id()
    print(id2config[incumbent])


@hydra.main(config_path="../config", config_name="cfg_affordance")
def optim_vrenv(cfg):
    worker = False
    if(cfg.optim.n_workers > 1):
        worker = True
    model_name = cfg.model_name
    optimize(model_name, cfg,
             run_id=cfg.optim.run_id,
             nameserver=cfg.optim.nameserver,
             max_budget=cfg.optim.max_budget,
             min_budget=cfg.optim.min_budget,
             n_iterations=cfg.optim.n_iterations,
             n_workers=cfg.optim.n_workers,
             worker=worker)
    read_results("%s.pkl" % model_name)


if __name__ == "__main__":
    optim_vrenv()
