import gym
import hydra
import os
import sys
# current_dir = os.path.dirname(
# os.path.abspath(inspect.getfile(inspect.currentframe())))
# parent_dir = os.path.dirname(current_dir)
parent_dir = os.path.dirname(os.getcwd())
sys.path.insert(0, os.getcwd())
sys.path.insert(0, parent_dir)
sys.path.insert(0, parent_dir+"/VREnv/")
from sac_agent.sac import SAC
from utils.env_processing_wrapper import EnvWrapper
from omegaconf import OmegaConf

gym.envs.register(
     id='VREnv-v0',
     entry_point='VREnv.vr_env.envs.play_table_env:PlayTableSimEnv',
     max_episode_steps=200,
)


def load_cfg(cfg_path, cfg, optim_res):
    if(os.path.exists(cfg_path) and not optim_res):
        run_cfg = OmegaConf.load(cfg_path)
        net_cfg = run_cfg.agent.net_cfg
        img_obs = run_cfg.img_obs
        env_wrapper = run_cfg.env_wrapper
        agent_cfg = run_cfg.agent.hyperparameters
    else:
        run_cfg = cfg
        net_cfg = cfg.agent.net_cfg
        img_obs = cfg.img_obs
        env_wrapper = cfg.env_wrapper
        agent_cfg = cfg.agent.hyperparameters

    return run_cfg, net_cfg, img_obs, env_wrapper, agent_cfg


@hydra.main(config_path="../config", config_name="cfg_sac")
def hydra_evaluateVRenv(cfg):
    # Get hydra config from tested model and load it
    # important parameters are hidden_dim (defines the network)
    # img_obs and img_wrapper
    test_cfg = cfg.test_sac
    optim_res = cfg.test_sac.optim_res
    # Load saved config
    run_cfg, net_cfg, img_obs, env_wrapper, agent_cfg =\
        load_cfg(test_cfg.folder_name + ".hydra/config.yaml", cfg, optim_res)

    # Create evaluation environment and wrapper for the image in case there's
    # an image observation
    cfg.eval_env.task = run_cfg.task
    cfg.eval_env.rand_init_state = run_cfg.rand_init_state
    print(cfg.eval_env.task)
    print("Random initial state: %s" % cfg.eval_env.rand_init_state)
    print(OmegaConf.to_yaml(env_wrapper))
    eval_env = gym.make("VREnv-v0", **cfg.eval_env).env
    eval_env = EnvWrapper(eval_env, **env_wrapper)

    # Load model
    path = "%s/trained_models/%s.pth" % (
            test_cfg.folder_name,
            test_cfg.model_name)
    print(os.path.abspath(path))
    model = SAC(eval_env, img_obs=img_obs, net_cfg=net_cfg, **agent_cfg)
    success = model.load(path)
    if(success):
        model.evaluate(eval_env, **cfg.test_sac.eval_cfg)


if __name__ == "__main__":
    hydra_evaluateVRenv()
