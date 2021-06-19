import hydra
import gym
import os
import sys
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
    run_cfg.all_cameras = cfg.all_cameras

    max_ts = cfg.agent.learn_config.max_episode_length

    save_images = cfg.test.eval_cfg.save_images
    dirs = ["frames", "gripper_aff", "static_aff",
            "gripper_dirs", "static_dirs", "static_centers"]
    if(save_images):
        for d in dirs:
            os.makedirs("./%s/" % d)
    env = gym.make("VREnv-v0", **run_cfg.eval_env).env
    env = wrap_env(env, max_ts,
                   affordance=run_cfg.affordance,
                   save_images=save_images,
                   **env_wrapper)

    sac_cfg = {"env": env,
               "model_name": run_cfg.model_name,
               "save_dir": run_cfg.agent.save_dir,
               "net_cfg": net_cfg,
               **agent_cfg}

    model = Combined(run_cfg, sac_cfg=sac_cfg)
    path = "%s/trained_models/%s.pth" % (
            test_cfg.folder_name,
            test_cfg.model_name)
    success = model.load(path)

    if(success):
        # model.evaluate(env, **cfg.test.eval_cfg)
        model.tidy_up(env)
    env.close()


if __name__ == "__main__":
    main()
