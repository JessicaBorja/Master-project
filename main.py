import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder
# import pybullet as p
# import pybullet_data
# from pybullet_envs.gym_manipulator_envs import ReacherBulletEnv, PusherBulletEnv
# from pybullet_envs.gym_locomotion_envs import HalfCheetahBulletEnv
# from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv
from gym.envs.mujoco import mujoco_env
import mujoco_py

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

def train():
    #env = gym.make("Pendulum-v0")
    #env = gym.make("ReacherBulletEnv-v0")
    #env_name = "HalfCheetahBulletEnv-v0"
    env_name = "Pusher-v2"
    hyperparameters["env"] = gym.make(env_name).env
    hyperparameters["eval_env"] = gym.make(env_name).env
    hyperparameters["model_name"] = "sac_mujocoPusher"
    model = SAC(**hyperparameters)
    model.learn(**learn_configuration)

def evaluate(eval_config, model_name):
    env_name = "Pusher-v2"
    env = gym.make(env_name).env

    model = SAC(env, hidden_dim=470)
    success = model.load("./trained_models/%s.pth"%model_name)
    if(success):
        model_name = model_name.split("_r-")[0]#take model name, remove reward
        eval_config["model_name"] = model_name
        model.evaluate(env, **eval_config)

def read_optim_results(name):
    with open(os.path.join("./optimization_results/", name), 'rb') as fh:
        res = pickle.load(fh)

    id2config = res.get_id2config_mapping()
    incumbent = res.get_incumbent_id()
    print(id2config[incumbent])

if __name__ == "__main__":
    #train()
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