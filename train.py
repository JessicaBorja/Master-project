import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder
# import pybullet as p
# import pybullet_data
# from pybullet_envs.gym_manipulator_envs import ReacherBulletEnv, PusherBulletEnv
# from pybullet_envs.gym_locomotion_envs import HalfCheetahBulletEnv
# from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv
# from gym.envs.mujoco import mujoco_env
# import mujoco_py
from sac import SAC
import hydra

@hydra.main(config_path="./config", config_name="config_gym")
def train(cfg):
    env_name = "Pendulum-v0"
    hyperparameters = cfg.agent.hyperparameter
    hyperparameters["env"] = gym.make(env_name).env
    hyperparameters["eval_env"] = gym.make(env_name).env
    model = SAC(**hyperparameters)
    model.learn(**cfg.agent.learn_config)
    
if __name__ == "__main__":
    train()
