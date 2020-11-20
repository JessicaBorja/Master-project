import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder
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
    cfg = yaml.load(open(path,'r'))
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

    model = SAC(eval_env, **agent_cfg)
    success = model.load(path)
    if(success):
        eval_config["model_name"] = model_name
        model.evaluate(eval_env, **eval_config)

if __name__ == "__main__":
    #read_optim_results("sac_mujocoPusher2.pkl")
    eval_config = {
        "n_episodes": 20,
        "render": True,
        "print_all_episodes": True,
        "write_file": False,
        "max_episode_length": 100,
    }
    model_name = "sac_VREnv_20-11_02-04_best_eval"
    folder_name = "2020-11-20/02-04-09"
    evaluateVRenv(eval_config, model_name, folder_name)