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

hyperparameters = {
    "gamma": 0.99,
    "actor_lr": 1e-5, 
    "critic_lr": 0.00035, 
    "alpha_lr": 1e-5,
    "alpha": "auto",#"auto",
    "tau": 0.008,
    "batch_size": 220,
    "buffer_size": 1e6,
    "train_freq": 1, #timesteps collectng data
    "gradient_steps": 1, #timesteps updating gradients
    "learning_starts": 1000, #timesteps before starting updates
    "hidden_dim": 470
}

learn_configuration = {
    "total_timesteps": int(1e6), #millon 1e6
    "log_interval": 1000, #log timestep reward every log_interval steps
    "max_episode_length": 200, #max episode length
}

def train(env_name):
    hyperparameters["env"] = gym.make(env_name).env
    hyperparameters["eval_env"] = gym.make(env_name).env
    hyperparameters["model_name"] = "sac_mujocoPusher"
    model = SAC(**hyperparameters)
    model.learn(**learn_configuration)
    
if __name__ == "__main__":
    env_name = "Pusher-v2"
    train(env_name)
    #read_optim_results("sac_mujocoPusher2.pkl")