import time
import gym
# import pybullet as p
# import pybullet_data
# from pybullet_envs.gym_manipulator_envs import ReacherBulletEnv, PusherBulletEnv
# from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv
from stable_baselines3.sac import SAC
from stable_baselines3.sac import MlpPolicy
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from datetime import datetime
from gym.wrappers.time_limit import TimeLimit
import time

hyperparameters = {
    "gamma": 0.99,
    "learning_rate": 3e-4,
    #"ent_coef": 1, auto
    "tau": 0.005,
    "target_update_interval":1,
    "batch_size": 256,
    "buffer_size": 50000,
    "train_freq": 1,
    "gradient_steps": 1,
    "policy_kwargs": dict(net_arch=[256, 256])
}

learn_configuration = {
    "total_timesteps": int(2e5), #millon 1e6
    "log_interval": 1, #log reward every log_interval episodes
}

def main():
    #env = KukaGymEnv(renders=True, isDiscrete=False)
    #env = ReacherBulletEnv(render = True)
    #env = PusherBulletEnv(render = True)
    #env = gym.make("Pendulum-v0")
    #env = gym.make("ReacherBulletEnv-v0")
    #env = gym.make("PusherBulletEnv-v0")
    
    #model
    env_name = "Pusher-v2"
    #env_name = "ReacherBulletEnv-v0"
    env = gym.make(env_name)
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./logs/')
    # Separate evaluation env
    eval_env = gym.make(env_name)#alreay has time limit set to 150
    eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/sb_reacher_best_model',
                                n_eval_episodes = 10, log_path='./logs/sb_reacher_results', eval_freq=1000)#eval_freq == steps

    model = SAC('MlpPolicy', env, tensorboard_log="./results/", verbose=1, **hyperparameters)
    model.learn(callback=eval_callback, tb_log_name="sb_sac_%s"%env_name[:-3],
                eval_log_path ="sb_reacher_eval", **learn_configuration)

if __name__ == "__main__":
    main()

#action space: -1.0, 1.0, 7
#obs space: -inf, inf, 55