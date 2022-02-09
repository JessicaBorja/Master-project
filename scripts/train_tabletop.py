import hydra
import logging
import os

from vapo.wrappers.play_table_rl import PlayTableRL
from vapo.wrappers.affordance.aff_wrapper_sim import AffordanceWrapperSim
from vapo.wrappers.utils import get_name
from vapo.agent.vapo_agent import VAPOAgent
from omegaconf import OmegaConf
import glob


@hydra.main(config_path="../config", config_name="cfg_tabletop")
def main(cfg):
    # Auto generate names given dense, aff-mask, aff-target
    previous_model = glob.glob("./trained_models/*last.pth")
    resume_training = len(previous_model) > 0
    if resume_training:
        model_path = previous_model[0]
        hydra_run_dir = os.getcwd()
        hydra_cfg_path = os.path.join(hydra_run_dir, ".hydra/config.yaml")
        print("Old configuration found, resuming training from same directory")
        if os.path.exists(hydra_cfg_path):
            cfg = OmegaConf.load(hydra_cfg_path)
    else:
        print("Starting new training")

    log = logging.getLogger(__name__)
    cfg.model_name = get_name(cfg, cfg.model_name)
    max_ts = cfg.agent.learn_config.max_episode_length
    env = PlayTableRL(**cfg.env)
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
    model = VAPOAgent(cfg,
                      sac_cfg=sac_cfg,
                      wandb_login=cfg.wandb_login,
                      resume=resume_training)
    if(resume_training):
        model.load(model_path,
                   resume_training=True)
    model.learn(**cfg.agent.learn_config)
    training_env.close()


if __name__ == "__main__":
    main()
