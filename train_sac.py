import gym
import pybullet as p
import hydra
from omegaconf import OmegaConf
import os,sys
parent_dir = os.path.dirname(os.getcwd())
sys.path.insert(0, parent_dir)
sys.path.insert(0, parent_dir+"/VREnv/")
gym.envs.register(
     id='VREnv-v0',
     entry_point='VREnv.src.envs.play_table_env:PlayTableSimEnv',
     max_episode_steps=200,
)
from utils.env_processing_wrapper import EnvWrapper
from sac_agent.sac import SAC
@hydra.main(config_path="./config", config_name="cfg_sac")
def main(cfg):
    print("agent configuration")
    print( OmegaConf.to_yaml(cfg.agent) )
    print( OmegaConf.to_yaml(cfg.env_wrapper) )
    print("repeat_training:%d, img_obs:%s"%(cfg.repeat_training, str(cfg.img_obs)))

    for i in range(cfg.repeat_training):
        training_env =  gym.make("VREnv-v0", **cfg.env).env
        eval_env =  gym.make("VREnv-v0", **cfg.eval_env).env
        training_env = EnvWrapper(training_env, train = True, **cfg.env_wrapper)
        eval_env =  EnvWrapper(eval_env, **cfg.env_wrapper)
        model_name = cfg.model_name
        model = SAC(env = training_env, eval_env = eval_env, model_name = model_name,\
                    save_dir = cfg.agent.save_dir, img_obs = cfg.img_obs, net_cfg=cfg.agent.net_cfg,
                    **cfg.agent.hyperparameters)

        model.learn(**cfg.agent.learn_config)
        training_env.close()
        eval_env.close()
        
if __name__ == "__main__":
    main()