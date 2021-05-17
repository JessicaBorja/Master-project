import hydra
import gym
import os
import sys
import pybullet as p
parent_dir = os.path.dirname(os.getcwd())
sys.path.insert(0, os.getcwd())
sys.path.insert(0, parent_dir)
sys.path.insert(0, parent_dir+"/VREnv/")
from env_wrappers.env_wrapper import wrap_env
from combined.combined import Combined
from utils.utils import load_cfg, register_env
register_env()


@hydra.main(config_path="../config", config_name="cfg_combined")
def main(cfg):
    test_cfg = cfg.test
    run_cfg, net_cfg, env_wrapper, agent_cfg =\
        load_cfg(os.path.join(test_cfg.folder_name, ".hydra/config.yaml"),
                 cfg, optim_res=False)

    run_cfg.eval_env.show_gui = cfg.env.show_gui
    run_cfg.eval_env.cameras = cfg.env.cameras

    env = gym.make("VREnv-v0", **run_cfg.eval_env).env
    env = wrap_env(env,
                   affordance=run_cfg.affordance,
                   **env_wrapper)

    sac_cfg = {"env": env,
               "model_name": run_cfg.model_name,
               "save_dir": run_cfg.agent.save_dir,
               "net_cfg": net_cfg,
               **agent_cfg}

    model = Combined(run_cfg, sac_cfg=sac_cfg)
    for i in range(50000):
        a_center, target = model._aff_compute_target()
        p.removeAllUserDebugItems()
        p.addUserDebugText("target_0",
                           textPosition=target,
                           textColorRGB=[0, 1, 0])
        p.addUserDebugText("a_center",
                           textPosition=a_center,
                           textColorRGB=[0, 1, 0])
    env.close()


if __name__ == "__main__":
    main()
