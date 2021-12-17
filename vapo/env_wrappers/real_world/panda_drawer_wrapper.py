import time

import gym
import numpy as np
from robot_io.utils.utils import pos_orn_to_matrix
import logging
import math

log = logging.getLogger(__name__)


class PandaEnvWrapper(gym.Wrapper):
    def __init__(self, env, d_pos, d_rot,
                 gripper_success_width,
                 gripper_success_displacement,
                 reward_fail, reward_success,
                 termination_radius, offset,
                 gripped_success_displacement, *args, **kwargs):
        super().__init__(env)
        self.d_pos = d_pos
        self.d_rot = d_rot
        self.gripper_success_width = gripper_success_width
        self.reward_fail = reward_fail
        self.reward_success = reward_success
        self.termination_radius = termination_radius
        self.offset = np.array([*offset, 1])
        self.gripper_success_displacement = \
            np.array([*gripper_success_displacement, 1])
        self._task = "drawer"
        # self._target_orn = np.array([- math.pi * 3/4, 0, 0])
        # Orn from cluster data
        self._target_orn = np.array([-2.26, 0.05, -0.12])

        # To track success of episode
        self.target_pos = None
        self.gripped_success_displacement = gripped_success_displacement

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
            tcp_mat = pos_orn_to_matrix(target_pos, target_orn)
            # Offset in gripper frame
            offset_global_frame = tcp_mat @ self.offset
            move_to = offset_global_frame[:3]
        else:
            move_to = target_pos
        return self.transform_obs(
                    self.env.reset(move_to, target_orn))

    def check_success(self, robot_obs):
        gripper_width = robot_obs["gripper_opening_width"]
        holding_obj = gripper_width > self.gripper_success_width

        tcp_pos = robot_obs["tcp_pos"]
        tcp_orn = robot_obs["tcp_orn"]
        T_world_tcp = pos_orn_to_matrix(tcp_pos, tcp_orn)

        # Initial position in end effector frame coords
        rel_disp_from_aff = - np.linalg.inv(T_world_tcp) @ self.target_pos
        moved_thresh_dist = rel_disp_from_aff[-1] >= self.gripped_success_displacement[-1]

        return holding_obj and moved_thresh_dist

    def check_termination(self, current_pos):
        return np.linalg.norm(self.target_pos - current_pos) > self.termination_radius

    def step(self, action, move_to_box=False):
        assert len(action) == 4

        rel_target_pos = np.array(action[:3]) * self.d_pos
        rel_target_orn = np.array([0, 0, 0])
        gripper_action = action[-1]

        action = {"motion": (rel_target_pos, rel_target_orn, gripper_action),
                  "ref": "rel"}

        obs, reward, done, info = self.env.step(action)
        success = False

        # On gripper close
        if gripper_action == -1:
            if self.check_success(obs["robot_state"]):
                reward = self.reward_success
                success = True
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

        info["success"] = success
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
