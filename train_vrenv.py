import gym
from sac import SAC
import sys,os
import pybullet as p
import time
import math
import hydra
import os,sys,inspect
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

@hydra.main(config_path="./config", config_name="config_rl")
def main(cfg):
    training_env =  gym.make("VREnv-v0", **cfg.env).env
    eval_env =  gym.make("VREnv-v0", **cfg.eval_env).env
    #training_env = hydra.utils.instantiate(cfg.env)
    #eval_env = hydra.utils.instantiate(cfg.eval_env)
    
    model_name = "sac_VREnv"
    model = SAC(env = training_env, eval_env = eval_env, model_name = model_name,\
                save_dir = cfg.agent.save_dir, **cfg.agent.hyperparameters)
    model.learn(**cfg.agent.learn_configuration)

    #p.resetDebugVisualizerCamera(cameraDistance=1.4, cameraYaw=-45, cameraPitch=-45, cameraTargetPosition=[-1, 0,0.5])
    #fixed_angle = p.getQuaternionFromEuler([-math.pi/2, -math.pi/2, 0])
    #action = ([0.5 , 0.83, 0.59], fixed_angle, 1)

    # model_name = "sac_VREnv_18-11_09-10_best_eval"
    # folder_name = "2020-11-18/09-10-28"
    # path = "./hydra_outputs/%s/trained_models/%s.pth"%(folder_name, model_name)
    # abspath = os.path.abspath(path).split
    # evaluate(eval_env, eval_config, model_name, abspath)

def evaluate(env, eval_config, model_name, path):
    model = SAC(env)
    success = model.load(path)
    if(success):
        eval_config["model_name"] = model_name
        model.evaluate(env, **eval_config)

if __name__ == "__main__":
    main()