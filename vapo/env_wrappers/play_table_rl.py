import os
import logging
import math
import numpy as np
import gym
import torch
import pybullet as p

from gym import spaces
from vr_env.envs.play_table_env import PlayTableSimEnv
from vr_env.utils.utils import EglDeviceNotFoundError, get_egl_device_id
from vapo.env_wrappers.play_table_rand_scene import PlayTableRandScene
logger = logging.getLogger(__name__)


class PlayTableRL(PlayTableSimEnv):
    def __init__(self, task="slide", sparse_reward=False, **args):
        if('use_egl' in args and args['use_egl']):
            # if("CUDA_VISIBLE_DEVICES" in os.environ):
            #     device_id = os.environ["CUDA_VISIBLE_DEVICES"]
            #     device = int(device_id)
            # else:
            device = torch.cuda.current_device()
            device = torch.device(device)
            self.set_egl_device(device)
        super(PlayTableRL, self).__init__(**args)
        self.task = task
        self._action_space_scale = np.array(
                    [0.03, 0.03, 0.03,
                     math.pi / 8, math.pi / 8, math.pi / 8, 1])
        _action_space = np.ones(7)
        self.action_space = spaces.Box(_action_space * -1, _action_space)
        obs_space_dict = {
            "scene_obs": gym.spaces.Box(low=0, high=1.5, shape=(3,)),
            'robot_obs': gym.spaces.Box(low=-0.5, high=0.5, shape=(7,)),
            'rgb_obs': gym.spaces.Box(low=0, high=255, shape=(3, 300, 300)),
            'depth_obs': gym.spaces.Box(low=0, high=255, shape=(1, 300, 300))
        }
        self.observation_space = gym.spaces.Dict(obs_space_dict)
        self.sparse_reward = sparse_reward
        self._action_space_scale = np.array(
                    [0.03, 0.03, 0.03,
                     math.pi / 8, math.pi / 8, math.pi / 8, 1])
        _action_space = np.ones(7)
        self.action_space = spaces.Box(_action_space * -1, _action_space)
        obs_space_dict = {
            "scene_obs": gym.spaces.Box(low=0, high=1.5, shape=(3,)),
            'robot_obs': gym.spaces.Box(low=-0.5, high=0.5, shape=(7,)),
            'rgb_obs': gym.spaces.Box(low=0, high=255, shape=(3, 300, 300)),
            'depth_obs': gym.spaces.Box(low=0, high=255, shape=(1, 300, 300))
        }
        self.observation_space = gym.spaces.Dict(obs_space_dict)
        self.sparse_reward = sparse_reward
        self.scene = PlayTableRandScene(p=self.p, cid=self.cid,
                                        np_random=self.np_random,
                                        **args['scene_cfg'])
        self.rand_positions = self.scene.rand_positions
        self.load()
        if(task == "pickup"):
            self.target = self.scene.target
            self.box_pos = self.scene.object_cfg['fixed_objects']['bin']["initial_pos"]

    def pick_rand_obj(self, p_dist=None):
        self.scene.pick_rand_obj(p_dist=p_dist)
        self.target = self.scene.target

    def load_scene_with_objects(self, obj_lst, load_scene=False):
        self.scene.pick_rand_obj(obj_lst, load_scene)
        self.target = self.scene.target

    def load_rand_scene(self, replace_objs=None, load_scene=False):
        self.scene.load_rand_scene(replace_objs, load_scene)
        self.target = self.scene.target

    def reset(self):
        res = super(PlayTableRL, self).reset()
        if(self.task == "pickup"):
            self.target = self.scene.target
        return res

    def set_egl_device(self, device):
        assert "EGL_VISIBLE_DEVICES" not in os.environ, "Do not manually set EGL_VISIBLE_DEVICES"
        cuda_id = device.index if device.type == "cuda" else 0
        try:
            egl_id = get_egl_device_id(cuda_id)
        except EglDeviceNotFoundError:
            logger.warning(
                "Couldn't find correct EGL device. Setting EGL_VISIBLE_DEVICE=0. "
                "When using DDP with many GPUs this can lead to OOM errors. "
                "Did you install PyBullet correctly? Please refer to VREnv README"
            )
            egl_id = 0
        os.environ["EGL_VISIBLE_DEVICES"] = str(egl_id)
        logger.info(f"EGL_DEVICE_ID {egl_id} <==> CUDA_DEVICE_ID {cuda_id}")

    def step(self, action):
        # Action space that SAC sees is between -1,1 for all elements in vector
        if(len(action) == 5):
            a = action.copy()
            if(self.task == "pickup"):  # Constraint angle
                a = [*a[:3], 0, 0, a[-2], a[-1]]
                # Scale vector to true values that it can take
                a = self.robot.relative_to_absolute(a)
                a = list(a)
                # constraint angle
                a[1] = np.array([- math.pi, 0, a[1][-1]])
        else:
            a = action
        self.robot.apply_action(a)
        for i in range(self.action_repeat):
            self.p.stepSimulation(physicsClientId=self.cid)
        self.scene.step()
        # dict w/keys: "rgb_obs", "depth_obs", "robot_obs","scene_obs"
        obs = self.get_obs()
        reward, r_info = self._reward()
        done = self._termination()
        info = self.get_info()
        info.update(r_info)
        # obs, reward, done, info
        return obs, reward, done, info

    def _normalize(self, val, min_val, max_val):
        return (val - min_val) / (max_val - min_val)

    def _reward(self):
        # returns the normalized state
        targetWorldPos, targetState = self.get_target_pos()
        # Only get end effector position
        robotObs, _ = self.robot.get_observation()
        robotPos = robotObs[:3]

        # Compute reward
        reward_state = targetState - 1  # 0 or 1

        # Normalize dist
        # dist = 1 if dist > 1 else dist
        info = {"reward_state": reward_state}
        if(self.sparse_reward):
            # 1 if lifted, else 0
            if self.task == "banana" or self.task == "pickup":
                reward = reward_state * 200  # 200 if pickup
                # If episode ended because it moved to far away
                if(self._termination() and not reward_state):
                    reward = -1
            elif self.task == "banana_combined":
                # If robot is more than 10 cm away from target, -1 reward
                # 2 if above table
                reward = -1 if self._termination() else reward_state * 2
            else:
                reward = self._termination() * 200  # 0 or 10
        else:
            reward_near = - np.linalg.norm(targetWorldPos - robotPos)
            reward = reward_near + reward_state
            info = {"reward_state": reward_state, "reward_near": reward_near}
        return reward, info

    def _termination(self):
        done = False
        targetPos, targetState = self.get_target_pos()
        if self.task == "slide":
            if(targetState > 0.7):
                done = True
        elif self.task == "hinge":
            if(targetState > 0.75):
                done = True
        elif self.task == "drawer":  # self.task == "drawer":
            if(targetState > 0.8):
                done = True
        elif self.task == "banana_combined":
            banana_floor = True if targetPos[2] < 0.4 else False
            done = banana_floor
        else:  # Task == pickup
            # 2 if lifted else 1
            done = targetState == 2
        # Return false if task not defined
        return done

    def get_target_pos(self):
        if self.task == "slide":
            link_id = self.scene.get_info()['fixed_objects']['table']['uid']
            targetWorldPos = self.p.getLinkState(link_id, 2,
                                                 physicsClientId=self.cid)[0]
            targetState = self.p.getJointState(link_id, 2,
                                               physicsClientId=self.cid)[0]

            # only keep x dim
            targetWorldPos = [targetWorldPos[0] - 0.1, 0.75, 0.74]
            targetState = self._normalize(targetState, 0, 0.56)
        elif self.task == "hinge":
            link_id = \
                self.scene.get_info()['fixed_objects']['hinged_drawer']['uid']
            targetWorldPos = self.p.getLinkState(link_id, 1,
                                                 physicsClientId=self.cid)[0]
            targetState = self.p.getJointState(link_id, 1,
                                               physicsClientId=self.cid)[0]

            targetWorldPos = [targetWorldPos[0] + 0.02, targetWorldPos[1], 1]
            # table is id 0,
            # hinge door state(0) increases as it moves to left 0 to 1.74
            targetState = self.p.getJointState(0, 0,
                                               physicsClientId=self.cid)[0]
            targetState = self._normalize(targetState, 0, 1.74) - 1
        elif self.task == "drawer":  # self.task == "drawer":
            link_id = self.scene.get_info()['fixed_objects'][self.task]['uid']
            targetWorldPos = self.p.getLinkState(link_id, 0,
                                                 physicsClientId=self.cid)[0]
            targetState = self.p.getJointState(link_id, 0,
                                               physicsClientId=self.cid)[0]

            targetWorldPos = [-0.05, targetWorldPos[1] - 0.38, 0.469]
            targetState = self._normalize(targetState, 0, 0.23)
        else:
            lifted = False
            # If any object lifted targetState = 2
            for name in self.scene.table_objs:
                target_obj = \
                    self.scene.get_info()['movable_objects'][name]
                base_pos = p.getBasePositionAndOrientation(
                                target_obj["uid"],
                                physicsClientId=self.cid)[0]
                # if(p.getNumJoints(target_obj["uid"]) == 0):
                #     pos = base_pos
                # else:
                #     pos = p.getLinkState(target_obj["uid"], 0)[0]

                # self.p.addUserDebugText("O", textPosition=pos,
                #                         textColorRGB=[0, 0, 1])
                # 2.5cm above initial position and object not already in box
                if(base_pos[-1] >= target_obj["initial_pos"][-1] + 0.025
                   and not self.obj_in_box(self.target)):
                    lifted = True
            targetState = 2 if lifted else 1
            # Return position of current target for training
            curr_target_uid = \
                self.scene.get_info()['movable_objects'][self.target]["uid"]
            if(p.getNumJoints(curr_target_uid) == 0):
                targetWorldPos = p.getBasePositionAndOrientation(
                    curr_target_uid,
                    physicsClientId=self.cid)[0]
            else:
                targetWorldPos = p.getLinkState(curr_target_uid, 0)[0]
        return targetWorldPos, targetState  # normalized

    def obj_in_box(self, obj_name):
        box_pos = self.box_pos
        obj_uid = self.scene.get_info()['movable_objects'][obj_name]['uid']
        targetPos = p.getBasePositionAndOrientation(
            obj_uid,
            physicsClientId=self.cid)[0]
        # x range
        x_range, y_range = False, False
        if(targetPos[0] > box_pos[0] - 0.12 and
           targetPos[0] <= box_pos[0] + 0.12):
            x_range = True
        if(targetPos[1] > box_pos[1] - 0.2 and
           targetPos[1] <= box_pos[1] + 0.2):
            y_range = True
        return x_range and y_range

    # DEBUG
    def _printJoints(self):
        for i in range(self.object_num):
            print("object%d" % i)
            for j in range(self.p.getNumJoints(i, physicsClientId=self.cid)):
                joint_info = self.p.getJointInfo(i, j,
                                                 physicsClientId=self.cid)
                print(joint_info)

    def step_debug(self):
        # ######### Debug link positions ################
        for i in range(self.p.getNumJoints(0)):  # 0 is table
            s = self.p.getLinkState(0, i)
            # print(p.getJointInfo(0,i))
            self.p.addUserDebugText(str(i),
                                    textPosition=s[0],
                                    textColorRGB=[1, 0, 0])
        # ######### Debug link positions ################
        self.p.stepSimulation()
