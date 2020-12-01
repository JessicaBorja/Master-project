import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import hydra
import pybullet as p
import pybullet_data
# from pybullet_envs.gym_manipulator_envs import ReacherBulletEnv, PusherBulletEnv
# from pybullet_envs.gym_locomotion_envs import HalfCheetahBulletEnv
# from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv
#from gym.envs.mujoco import mujoco_env
#import mujoco_py
from utils.utils import read_optim_results
from sac import SAC
from datetime import datetime
import time, os, pickle, sys, inspect
import yaml
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 
sys.path.insert(0, parent_dir+"/VREnv/") 
gym.envs.register(
     id='VREnv-v0',
     entry_point='VREnv.src.envs.play_table_env:PlayTableSimEnv',
     max_episode_steps=200,
)

def evaluate(eval_config, model_name):
    env_name = "Pusher-v2"
    env = gym.make(env_name).env

    model = SAC(env, hidden_dim=470)
    success = model.load("./trained_models/%s.pth"%model_name)
    if(success):
        model_name = model_name.split("_r-")[0]#take model name, remove reward
        eval_config["model_name"] = model_name
        model.evaluate(env, **eval_config)

def load_config(model_name, folder_name):
    path = "./hydra_outputs/%s/.hydra/config.yaml"%(folder_name)
    path = os.path.abspath(path)
    cfg = yaml.load(open(path,'r'), Loader=yaml.FullLoader)
    eval_env_cfg = cfg["eval_env"]
    agent_cfg = cfg["agent"]["hyperparameters"]
    eval_env_cfg["cameras"].pop()
    eval_env_cfg["cameras"] = [cfg["static_camera"], cfg["gripper_camera"]]
    return eval_env_cfg, agent_cfg

def evaluateVRenv(eval_config, model_name, hydra_folderpath):
    evalenv_cfg, agent_cfg = load_config(model_name, hydra_folderpath)
    if(eval_config["render"]):
        evalenv_cfg["show_gui"]= True
    eval_env =  gym.make("VREnv-v0", **evalenv_cfg).env
    path = "./hydra_outputs/%s/trained_models/%s.pth"%(folder_name, model_name)
    agent_cfg["hidden_dim"] = 470
    print(agent_cfg)
    model = SAC(eval_env, **agent_cfg)
    success = model.load(path)
    if(success):
        eval_config["model_name"] = model_name
        model.evaluate(eval_env, **eval_config)

@hydra.main(config_path="./config", config_name="config_rl")
def hydra_evaluateVRenv(cfg):
    #model_name = "sac_vrenv_optim_22-11_02-52_best_eval"
    # model_name = "vrenv_optim_neg_state_reward_25-11_03-43_best_eval" #Last slide that worked :c
    # folder_name = "2020-11-25/15-42-56"
    #model_name = "vrenv_optim_neg_state_reward_29-11_03-34_best_eval"
    #folder_name = "2020-11-29/12-22-14"
    # model_name = "vrenv_optim_neg_state_reward_29-11_03-49_best_eval"
    # folder_name = "2020-11-29/13-11-31"
    model_name = "sac_slide_x5_01-12_05-18_best_eval"
    folder_name = "cluster/2020-11-30/16-15-06"

    agent_cfg = cfg.agent.hyperparameters
    eval_config =  cfg.eval_config
    eval_env =  gym.make("VREnv-v0", **cfg.eval_env).env
    path = "../../../outputs/%s/trained_models/%s.pth"%(folder_name, model_name)
    print(os.path.abspath(path))
    print(agent_cfg)
    model = SAC(eval_env, **agent_cfg)
    success = model.load(path)
    if(success):
        model.evaluate(eval_env, model_name = model_name, **eval_config)

if __name__ == "__main__":
    #read_optim_results("sac_mujocoPusher2.pkl")
    #evaluateVRenv(eval_config, model_name, folder_name)
    hydra_evaluateVRenv()