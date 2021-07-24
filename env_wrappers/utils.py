import gym
import cv2
import numpy as np
from omegaconf.omegaconf import OmegaConf
import torch
import os
import pybullet as p

from affordance_model.segmentator_centers import Segmentator
from affordance_model.datasets import get_transforms

from utils.img_utils import torch_to_numpy, visualize


def get_name(cfg, model_name):
    if(cfg.env_wrapper.gripper_cam.use_img):
        model_name += "_img"
    if(cfg.env_wrapper.gripper_cam.use_depth):
        model_name += "_depth"
    if(cfg.affordance.gripper_cam.target_in_obs):
        model_name += "_target"
    if(cfg.affordance.gripper_cam.use_distance):
        model_name += "_dist"
    if(cfg.affordance.gripper_cam.use):
        model_name += "_affMask"
    if(cfg.affordance.gripper_cam.densify_reward):
        model_name += "_dense"
    else:
        model_name += "_sparse"
    return model_name


def find_cam_ids(cameras):
    static_id, gripper_id, test_cam_id = 0, 1, None
    for i, cam in enumerate(cameras):
        if "gripper" in cam.name:
            gripper_id = i
        elif "static" in cam.name:
            static_id = i
        elif "test" in cam.name:
            test_cam_id = i
    return static_id, gripper_id, test_cam_id


def get_obs_space(affordance_cfg, gripper_cam_cfg, static_cam_cfg,
                  channels, img_size, use_robot_obs, task):
    cfg_dict = {
        "gripper": [gripper_cam_cfg, affordance_cfg.gripper_cam],
        "static": [static_cam_cfg, affordance_cfg.static_cam]
    }
    obs_space_dict = {}
    for cam_type, config in cfg_dict.items():
        env_obs_cfg, aff_cfg = config
        if(env_obs_cfg.use_img):
            obs_space_dict["%s_img_obs" % cam_type] = gym.spaces.Box(
                low=0, high=255,
                shape=(channels, img_size, img_size))
        if(env_obs_cfg.use_depth):
            obs_space_dict['%s_depth_obs' % cam_type] = gym.spaces.Box(
                low=0, high=255,
                shape=(1, img_size, img_size))
        if(aff_cfg.use):
            obs_space_dict['%s_aff' % cam_type] = gym.spaces.Box(
                low=0, high=1,
                shape=(1, img_size, img_size))
    if(use_robot_obs):
        # *tcp_pos(3), *tcp_euler(1), gripper_width, gripper_action(1),
        if(task == "pickup"):
            obs_space_dict['robot_obs'] = gym.spaces.Box(
                low=-0.5, high=0.5, shape=(6,))
        else:
            obs_space_dict['robot_obs'] = gym.spaces.Box(
                low=-0.5, high=0.5, shape=(8,))
    if(affordance_cfg.gripper_cam.target_in_obs):
        obs_space_dict['detected_target_pos'] = gym.spaces.Box(
            low=-1, high=1, shape=(3,))
    if(affordance_cfg.gripper_cam.use_distance):
        obs_space_dict['target_distance'] = gym.spaces.Box(
            low=-1, high=1, shape=(1,))
    return gym.spaces.Dict(obs_space_dict)


def init_aff_net(affordance_cfg, cam_str):
    aff_net = None
    if(affordance_cfg):
        if(affordance_cfg.static_cam.use
           and cam_str == "static"):
            path = affordance_cfg.static_cam.model_path
            # Configuration of the model
            hp = {**affordance_cfg.hyperparameters,
                  "hough_voting": affordance_cfg.static_cam.hough_voting}
            hp = OmegaConf.create(hp)

            # Create model
            if(os.path.exists(path)):
                aff_net = Segmentator.load_from_checkpoint(
                                    path,
                                    cfg=hp)
                aff_net.cuda()
                aff_net.eval()
                print("obs_wrapper: Static cam affordance model loaded")
            else:
                affordance_cfg = None
                path = os.path.abspath(path)
                raise TypeError("Path does not exist: %s" % path)
        elif(cam_str == "gripper" and
                (affordance_cfg.gripper_cam.use or
                 affordance_cfg.gripper_cam.densify_reward)):
            path = affordance_cfg.gripper_cam.model_path

            # Configuration of the model
            hp = {**affordance_cfg.hyperparameters,
                  "hough_voting": affordance_cfg.gripper_cam.hough_voting}
            hp = OmegaConf.create(hp)

            # Create model
            if(os.path.exists(path)):
                aff_net = Segmentator.load_from_checkpoint(path, cfg=hp)
                aff_net.cuda()
                aff_net.eval()
                print("obs_wrapper: gripper affordance model loaded")
            else:
                affordance_cfg = None
                path = os.path.abspath(path)
                raise TypeError("Path does not exist: %s" % path)
    return aff_net, affordance_cfg


def depth_preprocessing(frame, img_size):
    # obs is from 0-255, (img_size, img_size, 1)
    new_frame = cv2.resize(
            frame,
            (img_size, img_size),
            interpolation=cv2.INTER_AREA)
    # (1, img_size, img_size)
    new_frame = np.expand_dims(new_frame, axis=0)
    return new_frame


def get_transforms_and_shape(transforms_cfg, in_size, out_size=None):
    apply_transforms = get_transforms(transforms_cfg, img_size=out_size)
    test_tensor = torch.zeros((3, in_size, in_size))
    test_tensor = apply_transforms(test_tensor)
    # History length
    if(test_tensor.shape == 2):
        test_tensor = test_tensor.unsqueeze(0)
    return apply_transforms, test_tensor.shape


def img_preprocessing(frame, transforms):
    # obs is from 0-255, (img_size, img_size, 3)
    frame = torch.from_numpy(frame).permute(2, 0, 1)
    frame = transforms(frame)
    frame = frame.cpu().detach().numpy()

    # History length
    # if(frame.shape == 2):
    #     frame = np.expand_dims(frame, 0)  # (1, img_size, img_size)
    # c, w, h = frame.shape
    # self._total_frames = np.pad(self._total_frames, ((
    #     c, 0), (0, 0), (0, 0)), mode='constant')[:-c, :, :]
    # self._total_frames[0:c] = frame

    # self._cur_img_obs = [self._total_frames[i * c:(i * c) + c]
    #                      for i in self._indices]
    # self._cur_img_obs = np.concatenate(self._cur_img_obs, 0)
    return frame
