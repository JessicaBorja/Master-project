import gym
import numpy as np
from gym import Env


class TerminationWrapper(gym.Wrapper):
    def __init__(self, env: Env, use_aff_target) -> None:
        super().__init__(env)
        self.use_aff_target = use_aff_target

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, reward, self.termination(done, observation), info

    def termination(self, done, obs):
        # If distance between detected target and robot pos
        #  deviates more than target_radius
        if(self.use_aff_target):
            distance = np.linalg.norm(self.unwrapped.current_target
                                      - obs["robot_obs"][:3])
        else:
            # Real distance
            target_pos, _ = self.get_target_pos()
            distance = np.linalg.norm(target_pos
                                      - obs["robot_obs"][:3])
        return done or distance > self.target_radius