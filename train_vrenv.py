import gym
from sac import SAC
import sys,os
import pybullet as p
import time
import math
import hydra
import os,sys,inspect
from utils.env_img_wrapper import ImgWrapper
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 
sys.path.insert(0, parent_dir+"/VREnv/") 
gym.envs.register(
     id='VREnv-v0',
     entry_point='VREnv.src.envs.play_table_env:PlayTableSimEnv',
     max_episode_steps=200,
)

eval_config = {
        "n_episodes": 20,
        "render": True,
        "print_all_episodes": False,
        "write_file": False,
        "max_episode_length": 100,
}

@hydra.main(config_path="./config", config_name="config_vrenv")
def main(cfg):
    #training_env = hydra.utils.instantiate(cfg.env)
    #eval_env = hydra.utils.instantiate(cfg.eval_env)
    print("agent configuration")
    print(cfg.agent)
    
    for i in range(cfg.repeat_training):
        training_env =  gym.make("VREnv-v0", **cfg.env).env
        eval_env =  gym.make("VREnv-v0", **cfg.eval_env).env
        if(cfg.img_obs):
            training_env = ImgWrapper(training_env)
            eval_env =  ImgWrapper(eval_env)
        model_name = cfg.model_name
        model = SAC(env = training_env, eval_env = eval_env, model_name = model_name,\
                    save_dir = cfg.agent.save_dir, img_obs = cfg.img_obs, **cfg.agent.hyperparameters)
        model.learn(**cfg.agent.learn_config)
        training_env.close()
        eval_env.close()
        
if __name__ == "__main__":
    main()