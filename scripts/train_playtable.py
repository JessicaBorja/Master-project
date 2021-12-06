import hydra
import logging
import os

from vapo.env_wrappers.play_table_rl import PlayTableRL
from vapo.env_wrappers.affordance.aff_wrapper_sim import AffordanceWrapperSim
from vapo.env_wrappers.utils import get_name
from vapo.combined.combined import Combined


@hydra.main(config_path="../config", config_name="cfg_playtable")
def main(cfg):
    # Auto generate names given dense, aff-mask, aff-target
    log = logging.getLogger(__name__)
    cfg.model_name = get_name(cfg, cfg.model_name)
    max_ts = cfg.agent.learn_config.max_episode_length
    env = PlayTableRL(**cfg.env)
    for i in range(cfg.repeat_training):
        training_env = AffordanceWrapperSim(env, max_ts,
                                            train=True,
                                            affordance_cfg=cfg.affordance,
                                            viz=cfg.viz_obs,
                                            save_images=cfg.save_images,
                                            **cfg.env_wrapper)
        sac_cfg = {"env": training_env,
                   "eval_env": None,
                   "model_name": cfg.model_name,
                   "save_dir": cfg.agent.save_dir,
                   "net_cfg": cfg.agent.net_cfg,
                   "train_mean_n_ep": cfg.agent.train_mean_n_ep,
                   "save_replay_buffer": cfg.agent.save_replay_buffer,
                   "log": log,
                   **cfg.agent.hyperparameters}

        log.info("model: %s" % cfg.model_name)
        model = Combined(cfg,
                         sac_cfg=sac_cfg,
                         wandb_login=cfg.wandb_login)

        if cfg.resume_training:
            original_dir = hydra.utils.get_original_cwd()
            model_path = os.path.join(original_dir, cfg.resume_model_path)
            path = "%s/trained_models/%s.pth" % (model_path,
                                                 cfg.model_name + "_last")
            if(os.path.exists(path)):
                model.load(path)
            else:
                print("Model path does not exist: %s \n Training from start"
                      % os.path.abspath(path))
        model.learn(**cfg.agent.learn_config)
        training_env.close()


if __name__ == "__main__":
    main()
