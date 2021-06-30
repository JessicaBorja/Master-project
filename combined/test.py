import hydra
import gym
import os
import sys
parent_dir = os.path.dirname(os.getcwd())
sys.path.insert(0, os.getcwd())
sys.path.insert(0, parent_dir)
sys.path.insert(0, parent_dir+"/VREnv/")
from env_wrappers.env_wrapper import init_env, wrap_env
from combined import Combined
from utils.utils import load_cfg


@hydra.main(config_path="../config", config_name="cfg_combined")
def main(cfg):
    test_cfg = cfg.test
    run_cfg, net_cfg, env_wrapper, agent_cfg =\
        load_cfg(os.path.join(test_cfg.folder_name, ".hydra/config.yaml"),
                 cfg, optim_res=False)

    run_cfg.eval_env.show_gui = cfg.env.show_gui
    run_cfg.eval_env.cameras = cfg.env.cameras
    run_cfg.all_cameras = cfg.all_cameras
    run_cfg.eval_env.use_egl = cfg.env.use_egl
    run_cfg.scene = cfg.scene

    max_ts = cfg.agent.learn_config.max_episode_length

    save_images = cfg.test.eval_cfg.save_images
    # env = gym.make("VREnv-v0", **run_cfg.eval_env).env
    env = init_env(run_cfg.eval_env)
    env = wrap_env(env, max_ts,
                   affordance=run_cfg.affordance,
                   save_images=save_images,
                   viz=cfg.test.eval_cfg.viz,
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
        model.tidy_up(env)
        # model.evaluate(env, **cfg.test.eval_cfg)
    env.close()


if __name__ == "__main__":
    main()
