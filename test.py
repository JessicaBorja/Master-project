import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder
# import pybullet as p
# import pybullet_data
# from pybullet_envs.gym_manipulator_envs import ReacherBulletEnv, PusherBulletEnv
# from pybullet_envs.gym_locomotion_envs import HalfCheetahBulletEnv
# from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv
from gym.envs.mujoco import mujoco_env
import mujoco_py
from utils.utils import read_optim_results
from sac import SAC
from datetime import datetime
import time, os, pickle

def evaluate(eval_config, model_name):
    env_name = "Pusher-v2"
    env = gym.make(env_name).env

    model = SAC(env, hidden_dim=470)
    success = model.load("./trained_models/%s.pth"%model_name)
    if(success):
        model_name = model_name.split("_r-")[0]#take model name, remove reward
        eval_config["model_name"] = model_name
        model.evaluate(env, **eval_config)

if __name__ == "__main__":
    #read_optim_results("sac_mujocoPusher2.pkl")
    eval_config = {
        "n_episodes": 20,
        "render": True,
        "print_all_episodes": False,
        "write_file": False,
        "max_episode_length": 100,
    }
    model_name = "sac_mujocoPusher_10-11_07-46_best_eval"
    evaluate(eval_config, model_name)