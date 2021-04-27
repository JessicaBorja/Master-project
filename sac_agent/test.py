import gym
import hydra
import os
import sys
parent_dir = os.path.dirname(os.getcwd())
sys.path.insert(0, os.getcwd())
sys.path.insert(0, parent_dir)
sys.path.insert(0, parent_dir+"/VREnv/")
from sac_agent.sac import SAC
from env_wrappers.env_wrapper import ObservationWrapper
from utils.utils import load_cfg, register_env
register_env()


@hydra.main(config_path="../config", config_name="cfg_sac")
def hydra_evaluateVRenv(cfg):
    # Get hydra config from tested model and load it
    # important parameters are hidden_dim (defines the network)
    # img_obs and img_wrapper
    test_cfg = cfg.test
    optim_res = cfg.test.optim_res
    # Load saved config
    run_cfg, net_cfg, env_wrapper, agent_cfg =\
        load_cfg(os.path.join(test_cfg.folder_name, ".hydra/config.yaml"),
                 cfg, optim_res)

    # Create evaluation environment and wrapper for the image in case there's
    # an image observation
    run_cfg.eval_env.show_gui = cfg.eval_env.show_gui
    run_cfg.eval_env.cameras = cfg.camera_conf.cameras
    print(run_cfg.eval_env.task)
    print("Random initial state: %s" % run_cfg.eval_env.rand_init_state)
    eval_env = gym.make("VREnv-v0", **run_cfg.eval_env).env
    eval_env = ObservationWrapper(eval_env, **env_wrapper)

    # Load model
    path = "%s/trained_models/%s.pth" % (
            test_cfg.folder_name,
            test_cfg.model_name)
    print(os.path.abspath(path))
    model = SAC(eval_env, net_cfg=net_cfg, **agent_cfg)
    success = model.load(path)
    if(success):
        model.evaluate(eval_env, **cfg.test.eval_cfg)


if __name__ == "__main__":
    hydra_evaluateVRenv()
