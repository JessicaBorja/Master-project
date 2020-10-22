import pybullet as p
import time
import pybullet_data
import gym
from pybullet_envs.gym_manipulator_envs import ReacherBulletEnv, PusherBulletEnv
from pybullet_envs.gym_locomotion_envs import HalfCheetahBulletEnv
from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv
from sac import SAC
from datetime import datetime
import time

hyperparameters = {
    "gamma": 0.99,
    "learning_rate": 3e-4,
    "ent_coef": 1,
    "tau": 0.01,
    "target_update_interval":1,
    "batch_size": 256,
    "buffer_size": 1e6,
    "train_freq": 1, #timesteps collectng data
    "gradient_steps": 1, #timesteps updating gradients
    #Hard update:
    # "tau": 1,
    # "target_update_interval": 1000,
    # "grad_steps": 4,
}

learn_configuration = {
    "total_timesteps": int(2e6), #millon 1e6
    "log_interval": 100000, #log timestep reward every log_interval steps
    "max_episode_length": 1000, #max episode length
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
    env_name = "PusherBulletEnv-v0"
    hyperparameters["env"] = HalfCheetahBulletEnv(render = False) #gym.make(env_name)
    hyperparameters["eval_env"] = HalfCheetahBulletEnv(render = False)#gym.make(env_name)
    hyperparameters["model_name"] = "sac_halfCheetah"
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