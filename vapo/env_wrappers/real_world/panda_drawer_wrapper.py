import time

import gym
import numpy as np
from vapo.utils.utils import get_3D_end_points
from robot_io.utils.utils import quat_to_euler
import logging
import math

log = logging.getLogger(__name__)


class PandaEnvWrapper(gym.Wrapper):
    def __init__(self, env, d_pos, d_rot, gripper_success_threshold, reward_fail, reward_success, termination_radius, offset, *args, **kwargs):
        super().__init__(env)
        self.d_pos = d_pos
        self.d_rot = d_rot
        self.gripper_success_threshold = gripper_success_threshold
        self.reward_fail = reward_fail
        self.reward_success = reward_success
        self.termination_radius = termination_radius
        self.target_pos = None
        self.offset = offset["drawer"]
        self._task = "drawer"
        self._target_orn = np.array([- math.pi * 3/4, 0, 0])

    @property
    def task(self):
        return self._task

    @task.setter
    def task(self, value):
        self._task = value

    @property
    def target_orn(self):
        return self._target_orn

    def reset(self, target_pos=None, target_orn=None):
        self.env.robot.open_gripper()
        if target_pos is not None and target_orn is not None:
            self.target_pos = target_pos
            move_to = target_pos + self.offset
        else:
            move_to = target_pos
        return self.transform_obs(
                    self.env.reset(move_to, target_orn))

    def check_success(self):
        return False

    def check_termination(self, current_pos):
        return np.linalg.norm(self.target_pos - current_pos) > self.termination_radius

    def step(self, action, move_to_box=False):
        assert len(action) == 4

        rel_target_pos = np.array(action[:3]) * self.d_pos
        rel_target_orn = np.array([0, 0, 0])
        gripper_action = action[-1]

        curr_pos = self.env.robot.get_tcp_pos_orn()[0]
        depth_thresh = curr_pos[-1] <= self.env.workspace_limits[0][-1] + 0.01
        if(depth_thresh):
            print("depth tresh")
            gripper_action = -1

        action = {"motion": (rel_target_pos, rel_target_orn, gripper_action),
                  "ref": "rel"}

        obs, reward, done, info = self.env.step(action)

        info["success"] = False

        if gripper_action == -1:
            done = True
            if self.check_success():
                reward = self.reward_success
                info["success"] = True
            else:
                reward = self.reward_fail
                info["failure_case"] = "failed_open"
        else:
            done = self.check_termination(obs["robot_state"]["tcp_pos"])
            if done:
                reward = self.reward_fail
                info["failure_case"] = "outside_radius"
        if('failure_case' in info):
            print(info['failure_case'])

        obs = self.transform_obs(obs)
        return obs, reward, done, info

    def get_obs(self):
        return self.transform_obs(self.env._get_obs())

    @staticmethod
    def transform_obs(obs):
        robot_obs = obs['robot_state']
        obs['robot_obs'] = np.concatenate([robot_obs["tcp_pos"], [robot_obs["gripper_opening_width"]]])
        del obs["robot_state"]
        return obs
