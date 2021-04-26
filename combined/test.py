import hydra
import gym
from env_wrappers.env_wrapper import wrap_env
from utils.utils import register_env
from combined.combined import Combined
register_env()


@hydra.main(config_path="./config", config_name="cfg_combined")
def main(cfg):
    env = gym.make("VREnv-v0", **cfg.eval_env).env
    env = wrap_env(env,
                   affordance=cfg.affordance,
                   **cfg.env_wrapper)
    sac_cfg = {"env": env,
               "model_name": cfg.model_name,
               "save_dir": cfg.agent.save_dir,
               "net_cfg": cfg.agent.net_cfg,
               **cfg.agent.hyperparameters}

    model = Combined(cfg, sac_cfg=sac_cfg)
    model.evaluate(env, **cfg.test.eval_cfg)
    env.close()


if __name__ == "__main__":
    main()
