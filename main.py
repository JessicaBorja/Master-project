import time
import gym
# import pybullet as p
# import pybullet_data
# from pybullet_envs.gym_manipulator_envs import ReacherBulletEnv, PusherBulletEnv
# from pybullet_envs.gym_locomotion_envs import HalfCheetahBulletEnv
# from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv
from gym.envs.mujoco import mujoco_env
import mujoco_py

from sac import SAC
from datetime import datetime
import time

hyperparameters = {
    "gamma": 0.98,
    "actor_lr": 3e-4, 
    "critic_lr": 3e-4, 
    "alpha_lr": 1e-5,
    "ent_coef": "auto",
    "tau": 0.02,
    "batch_size": 256,
    "buffer_size": 1e6,
    "train_freq": 1, #timesteps collectng data
    "gradient_steps": 1, #timesteps updating gradients
    "learning_starts": 1000 #timesteps before starting updates
}

learn_configuration = {
    "total_timesteps": int(1e6), #millon 1e6
    "log_interval": 1000, #log timestep reward every log_interval steps
    #"max_episode_length": 1000, #max episode length
}

eval_config = {
    "max_episode_length": 1000, 
    "n_episodes": 100,
    "render": False,
    "print_all_episodes": False,
    "write_file": True,
}

def main():
    #env = gym.make("Pendulum-v0")
    #env = gym.make("ReacherBulletEnv-v0")
    #env_name = "HalfCheetahBulletEnv-v0"
    env_name = "Pusher-v2"
    hyperparameters["env"] = gym.make(env_name)
    hyperparameters["eval_env"] = gym.make(env_name)
    hyperparameters["model_name"] = "sac_mujocoPusher"
    model = SAC(**hyperparameters)
    model.learn(**learn_configuration)

    #Evaluation
    #eval_env = HalfCheetahBulletEnv()
    #env = HalfCheetahBulletEnv(renders = True)
    #env = gym.make(env_name)
    #model_name = "sac_halfCheetah_21-10_07-31_r-2516"
    #evaluate_trained(env, model_name, eval_config)

def evaluate_trained(env, model_name, eval_config):
    model = SAC(env)
    success = model.load("./trained_models/%s.pth"%model_name)
    if(success):
        model_name = model_name.split("_r-")[0]#take model name, remove reward
        eval_config["model_name"] = model_name
        model.evaluate(env, **eval_config)

if __name__ == "__main__":
    main()