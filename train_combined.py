import hydra
import gym
import logging

from env_wrappers.env_wrapper import wrap_env
from utils.utils import register_env
from combined.combined import Combined
from vr_env.envs.play_table_env import PlayTableSimEnv


def get_name(cfg, model_name):
    if(cfg.affordance.gripper_cam.target_in_obs):
        model_name += "_target"
    if(cfg.affordance.gripper_cam.use):
        model_name += "_affMask"
    if(cfg.affordance.gripper_cam.densify_reward):
        model_name += "_dense"
    else:
        model_name += "_sparse"
    return model_name


@hydra.main(config_path="./config", config_name="cfg_combined")
def main(cfg):
    # register_env()
    # Auto generate names given dense, aff-mask, aff-target
    log = logging.getLogger(__name__)
    cfg.model_name = get_name(cfg, cfg.model_name)
    max_ts = cfg.agent.learn_config.max_episode_length
    for i in range(cfg.repeat_training):
        # training_env = gym.make("VREnv-v0", **cfg.env).env
        training_env = PlayTableSimEnv(**cfg.env)
        training_env = wrap_env(training_env, max_ts,
                                train=True, affordance=cfg.affordance,
                                **cfg.env_wrapper)

        # eval_env = gym.make("VREnv-v0", **cfg.eval_env).env
        # eval_env = wrap_env(eval_env,
        #                     affordance=cfg.affordance,
        #                     **cfg.env_wrapper)

        sac_cfg = {"env": training_env,
                   "eval_env": None,
                   "model_name": cfg.model_name,
                   "save_dir": cfg.agent.save_dir,
                   "net_cfg": cfg.agent.net_cfg,
                   "log": log,
                   **cfg.agent.hyperparameters}

        log.info("model: %s" % cfg.model_name)
        model = Combined(cfg, sac_cfg=sac_cfg)
        model.learn(**cfg.agent.learn_config)
        training_env.close()
        # eval_env.close()


if __name__ == "__main__":
    main()
