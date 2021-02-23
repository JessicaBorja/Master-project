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


def check_consistency(cfg, run_cfg):
    # Default values in case not in config because it's old hydra structure
    img_obs = False
    env_wrapper = cfg.env_wrapper
    net_cfg = cfg.agent.net_cfg

    # find hidden dim
    try:
        net_cfg.hidden_dim = cfg.test_sac.hidden_dim
        print("hidden_dim loaded from test_sac.hidden_dim: %d" %
              net_cfg.hidden_dim)
    except Exception as e:
        if(run_cfg.agent.net_cfg):  # net structure
            net_cfg = run_cfg.agent.net_cfg
            print("hidden_dim loaded from agent.net_cfg.hidden_dim: %d" %
                  OmegaConf.to_yaml(net_cfg))
        elif(run_cfg.agent.hyperparameters.hidden_dim):
            net_cfg.hidden_dim = run_cfg.agent.hyperparameters.hidden_dim
            print("hidden_dim loaded from agent.hyperparmeters.hidden_dim:" +
                  "%d" % net_cfg.hidden_dim)
        else:  # no hidden_dim in run_cfg
            print("taking hidden_dim default value: %d" % net_cfg.hidden_dim)
        if(run_cfg.img_obs):
            img_obs = run_cfg.img_obs
        else:
            print("No img_obs in loaded config, taking default value: %s"
                  % img_obs)
            pass

    # Wrapper configuration
    if(run_cfg.env_wrapper):
        env_wrapper = run_cfg.env_wrapper
    elif(run_cfg.img_wrapper):
        env_wrapper = run_cfg.img_wrapper
    else:
        print("No env_wrapper in loaded config, taking default value:\n%s" %
              OmegaConf.to_yaml(cfg.env_wrapper))

    # Set env configurations
    try:
        cfg_lst = ["use_img", "use_pos", "use_depth"]
        for c in cfg_lst:
            env_wrapper[c] = net_cfg[c]
            net_cfg.pop(c)
    except Exception as e:
        pass

    return net_cfg, img_obs, env_wrapper


def load_cfg(cfg_path, cfg):
    if(not os.path.exists(cfg_path)):
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

    # Load saved config
    run_cfg, net_cfg, img_obs, env_wrapper, agent_cfg =\
        load_cfg(test_cfg.folder_name + ".hydra/config.yaml", cfg)

    # Create evaluation environment and wrapper for the image in case there's
    # an image observation
    cfg.eval_env.task = run_cfg.task
    print(cfg.eval_env.task)
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
