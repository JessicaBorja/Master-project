import hydra
import gym
from env_wrappers.env_wrapper import wrap_env
from utils.utils import register_env
from combined.combined import Combined
register_env()


@hydra.main(config_path="./config", config_name="cfg_combined")
def main(cfg):
    training_env = gym.make("VREnv-v0", **cfg.env).env
    eval_env = gym.make("VREnv-v0", **cfg.eval_env).env
    training_env = wrap_env(training_env, train=True,
                            affordance=cfg.affordance,
                            **cfg.env_wrapper)
    eval_env = wrap_env(eval_env,
                        affordance=cfg.affordance,
                        **cfg.env_wrapper)
    eval_env = None
    sac_cfg = {"env": training_env,
               "eval_env": eval_env,
               "model_name": cfg.model_name,
               "save_dir": cfg.agent.save_dir,
               "net_cfg": cfg.agent.net_cfg,
               **cfg.agent.hyperparameters}

    model = Combined(cfg, sac_cfg=sac_cfg)
    model.learn(**cfg.agent.learn_config)
    training_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
