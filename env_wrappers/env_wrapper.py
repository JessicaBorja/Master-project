import os
import logging
import torch
import numpy as np
import cv2
import pybullet as p

import gym
from gym import spaces

from utils.cam_projections import pixel2world
from vr_env.utils.utils import EglDeviceNotFoundError, get_egl_device_id
from utils.img_utils import torch_to_numpy, viz_aff_centers_preds, visualize
from .utils import get_obs_space, get_transforms_and_shape, find_cam_ids, \
                   init_aff_net, depth_preprocessing, img_preprocessing
logger = logging.getLogger(__name__)


def wrap_env(env_cfg, max_ts, save_images=False,
             viz=False, use_aff_target=False, **args):
    env = RLWrapper(env_cfg, max_ts,
                    save_images=save_images,
                    viz=viz,
                    **args)
    # env = RewardWrapper(env, max_ts)
    # env = TerminationWrapper(env, use_aff_target=use_aff_target)
    return env


class RLWrapper(gym.Wrapper):
    def __init__(self, EnvClass, env_cfg, max_ts, img_size,
                 gripper_cam, static_cam, transforms=None,
                 use_pos=False, use_aff_termination=False,
                 affordance_cfg=None, target_search="env",
                 max_target_dist=0.15,
                 train=False, save_images=False, viz=False,
                 history_length=None, skip_frames=None):
        # ENV definition
        if(env_cfg.use_egl):
            device = torch.device(torch.cuda.current_device())
            self.set_egl_device(device)
        self.env = EnvClass(**env_cfg)
        self.env.target_radius = max_target_dist
        self.initial_target_pos = None

        super(RLWrapper, self).__init__(self.env)
        self.task = self.env.task
        self.target_radius = self.env.target_radius

        # TERMINATION
        self.use_aff_termination = use_aff_termination or \
            target_search == "affordance"

        # REWARD FUNCTION
        self.affordance_cfg = affordance_cfg
        self.static_id, self.gripper_id, _ = find_cam_ids(self.env.cameras)
        self.ts_counter = 0
        self.max_ts = max_ts
        if(self.affordance_cfg.gripper_cam.densify_reward):
            print("RewardWrapper: Gripper cam to shape reward")

        # OBSERVATION
        self.img_size = img_size
        shape = (1, img_size, img_size)

        # Prepreocessing for affordance model
        _transforms_cfg = affordance_cfg.transforms["validation"]
        _static_aff_im_size = 200
        if(img_size in affordance_cfg.static_cam):
            _static_aff_im_size = affordance_cfg.static_cam.img_size
        self.aff_transforms = {
            "static": get_transforms_and_shape(_transforms_cfg,
                                               self.img_size,
                                               out_size=_static_aff_im_size)[0],
            "gripper": get_transforms_and_shape(_transforms_cfg,
                                                self.img_size)[0]}

        # Preprocessing for RL policy obs
        _transforms_cfg =\
            transforms["train"] if train else transforms["validation"]
        self.rl_transforms, shape = get_transforms_and_shape(_transforms_cfg,
                                                             self.img_size)
        self.channels = shape[0]

        # Cameras defaults
        self.obs_it = 0
        self.save_images = save_images
        self.viz = viz

        # Parameters to define observation
        self.gripper_cam_cfg = gripper_cam
        self.static_cam_cfg = static_cam
        self.use_robot_obs = use_pos
        # self._mask_transforms = DistanceTransform()

        # Parameters to store affordance
        self.gripper_cam_aff_net, affordance_cfg = \
            init_aff_net(affordance_cfg, 'gripper')
        self.static_cam_aff_net, affordance_cfg = \
            init_aff_net(affordance_cfg, 'static')
        self.curr_raw_obs = None

        # Observation and action space
        if(self.env.task == "pickup"):
            # 0-2 -> position , 3 -> yaw angle 4 gripper action
            _action_space = np.ones(5)
        else:
            # 0-2 -> position , 3-5 -> orientation, 6 gripper action
            _action_space = np.ones(7)
        self.action_space = spaces.Box(_action_space * -1, _action_space)
        self.observation_space = get_obs_space(affordance_cfg,
                                               self.gripper_cam_cfg,
                                               self.static_cam_cfg,
                                               self.channels, self.img_size,
                                               self.use_robot_obs,
                                               self.task)

        # Save images
        self.gripper_cam_imgs = {}

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

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self.initial_target_pos=None
        return self.observation(observation)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation), self.reward(reward), \
            self.termination(done, observation), info

    def observation(self, obs):
        # "rgb_obs", "depth_obs", "robot_obs","scene_obs"
        obs = {}
        self.curr_raw_obs = self.get_obs()
        obs_dict = self.curr_raw_obs
        obs = {**self.get_cam_obs(obs_dict, "gripper",
                                  self.gripper_cam_aff_net,
                                  self.gripper_cam_cfg,
                                  self.affordance_cfg.gripper_cam,
                                  self.gripper_id),
               **self.get_cam_obs(obs_dict, "static",
                                  self.static_cam_aff_net,
                                  self.static_cam_cfg,
                                  self.affordance_cfg.static_cam,
                                  self.static_id)}
        if(self.use_robot_obs):
            if(self.task == "pickup"):
                # *tcp_pos(3), *tcp_euler(1) z angle ,
                # gripper_opening_width(1), gripper_action
                obs["robot_obs"] = np.array([*obs_dict["robot_obs"][:3],
                                            *obs_dict["robot_obs"][5:7],
                                            obs_dict["robot_obs"][-1]])
            else:
                # *tcp_pos(3), *tcp_euler(3),
                # gripper_opening_width(1), gripper_action
                obs["robot_obs"] = np.array([*obs_dict["robot_obs"][:7],
                                            obs_dict["robot_obs"][-1]])
        self.obs_it += 1
        return obs

    def reward(self, rew):
        # modify rew
        if(self.affordance_cfg.gripper_cam.densify_reward):
            # set by observation wrapper so that
            # both have the same observation on
            # a given timestep
            if(self.curr_raw_obs is not None):
                obs_dict = self.curr_raw_obs
            else:
                obs_dict = self.get_obs()
            tcp_pos = obs_dict["robot_obs"][:3]

            # Create positive reward relative to the distance
            # between the closest point detected by the affordances
            # and the end effector position
            # p.addUserDebugText("target_reward",
            #                    textPosition=self.env.unwrapped.current_target,
            #                    textColorRGB=[0, 1, 0])

            # If episode is not done because of moving to far away
            if(not self.termination(self.env._termination(), obs_dict)
               and self.ts_counter <= self.max_ts):
                distance = np.linalg.norm(tcp_pos - self.env.unwrapped.current_target)
                # cannot be larger than 1
                # scale dist increases as it falls away from object
                scale_dist = min(distance / self.target_radius, 1)
                if(self.task == "pickup"):
                    rew += (1 - scale_dist)**0.4
                elif(self.task == "slide"):
                    goal_pose = np.array([0.079, 0.75, 0.74])
                    dist_to_goal = np.linalg.norm(self.env.unwrapped.current_target - goal_pose)
                    dist_to_goal /= 0.38  # max posible distance
                    rew -= scale_dist - dist_to_goal
                else:
                    rew -= scale_dist
                self.ts_counter += 1
            else:
                # If episode was successful
                if(rew >= 1 and self.env.task == "pickup"):
                    # Reward for remaining ts
                    rew += self.max_ts - self.ts_counter

                # If terminated because it went far away
                if(rew <= -1 and self.env.task != "pickup"):
                    # Penalize for remaining ts
                    # it would have gotten -1 for being far 
                    # and -1 for not completing task
                    rew -= (self.max_ts - self.ts_counter) * 2
                self.ts_counter = 0
        return rew

    def termination(self, done, obs):
        # If distance between detected target and robot pos
        #  deviates more than target_radius
        # p.removeAllUserDebugItems()
        # p.addUserDebugText("i",
        #                    textPosition=self.initial_target_pos,
        #                    textColorRGB=[0, 0, 1])
        # p.addUserDebugText("h",
        #                    textPosition=self.env.unwrapped.current_target,
        #                    textColorRGB=[0, 1, 0])
        if(self.use_aff_termination):
            distance = np.linalg.norm(self.initial_target_pos
                                      - obs["robot_obs"][:3])
        else:
            # Real distance
            target_pos, _ = self.env.get_target_pos()
            distance = np.linalg.norm(target_pos
                                      - obs["robot_obs"][:3])
        return done or distance > self.target_radius

    def get_cam_obs(self, obs_dict, cam_type, aff_net,
                    obs_cfg, aff_cfg, cam_id):
        obs = {}
        if(obs_cfg.use_depth):
            # Resize
            depth_obs = depth_preprocessing(
                            obs_dict['depth_obs'][cam_id],
                            self.img_size)
            obs["%s_depth_obs" % cam_type] = depth_obs
        if(obs_cfg.use_img):
            # Transform rgb to grayscale
            img_obs = img_preprocessing(
                                obs_dict['rgb_obs'][cam_id],
                                self.rl_transforms)
            # 1, W, H
            obs["%s_img_obs" % cam_type] = img_obs

        get_gripper_target = cam_type == "gripper" and (
                    self.affordance_cfg.gripper_cam.densify_reward
                    or self.affordance_cfg.gripper_cam.target_in_obs
                    or self.affordance_cfg.gripper_cam.use_distance)

        if(aff_net is not None and (aff_cfg.use or get_gripper_target)):
            # Np array 1, H, W
            processed_obs = img_preprocessing(
                                obs_dict['rgb_obs'][cam_id],
                                self.aff_transforms[cam_type])
            with torch.no_grad():
                # 1, 1, H, W in range [-1, 1]
                obs_t = torch.tensor(processed_obs).unsqueeze(0)
                obs_t = obs_t.float().cuda()

                # 1, H, W
                # aff_logits, aff_probs, aff_mask, directions
                _, aff_probs, aff_mask, directions = aff_net(obs_t)
                # aff_mask = self._mask_transforms(aff_mask).cuda()
                if(self.viz and cam_type == "static"):
                    visualize(aff_mask,
                              obs_dict['rgb_obs'][cam_id][:, :, ::-1],
                              imshow=True)
                mask = torch_to_numpy(aff_mask)  # foreground/affordance Mask       
                if(obs_cfg.use_img and aff_mask.shape[1:] != img_obs.shape[1:]):
                    new_shape = (aff_mask.shape[0], *img_obs.shape[1:])
                    mask = np.resize(mask, new_shape)
                if(get_gripper_target):
                    preds = {"%s_aff" % cam_type: aff_mask,
                             "%s_center_dir" % cam_type: directions,
                             "%s_aff_probs" % cam_type: aff_probs}

                    # Computes newest target
                    self.find_target_center(self.gripper_id,
                                            obs_dict['rgb_obs'][self.gripper_id],
                                            obs_dict['depth_obs'][self.gripper_id],
                                            obs_dict["robot_obs"][6],
                                            preds)
            if(self.affordance_cfg.gripper_cam.target_in_obs):
                obs["detected_target_pos"] = self.env.unwrapped.current_target
            if(self.affordance_cfg.gripper_cam.use_distance):
                distance = np.linalg.norm(self.env.unwrapped.current_target
                                          - obs_dict["robot_obs"][:3])
                obs["target_distance"] = np.array([distance])
            if(aff_cfg.use):
                # m = np.transpose(mask * 255,(1, 2, 0)).astype('uint8')
                # cv2.imshow("%s_aff" % cam_type, m)
                obs["%s_aff" % cam_type] = mask
        return obs

    # Aff-center
    def find_target_center(self, cam_id, orig_img, depth, gripper_width, obs):
        """
        Args:
            orig_img: np array, RGB original resolution from camera
                        shape = (1, cam.height, cam.width)
                        range = 0 to 255
                        Only used for vizualization purposes
            obs: dictionary
                - "img_obs":
                - "gripper_aff":
                    affordance segmentation mask, range 0-1
                    np.array(size=(1, img_size,img_size))
                - "gripper_aff_probs":
                    affordance activation function output
                    np.array(size=(1, n_classes, img_size,img_size))
                    range 0-1
                - "gripper_center_dir": center direction predictions
                    vectors in pixel space
                    np.array(size=(1, 2, img_size,img_size))
                np array, int64
                    shape = (1, img_size, img_size)
                    range = 0 to 1
        return:
            centers: list of 3d points (x, y, z)
        """
        aff_mask = obs["gripper_aff"]
        aff_probs = obs["gripper_aff_probs"]
        directions = obs["gripper_center_dir"]
        cam = self.cameras[cam_id]

        # Predict affordances and centers
        aff_mask, center_dir, object_centers, object_masks = \
            self.gripper_cam_aff_net.predict(aff_mask, directions)

        # Visualize predictions
        if(self.viz or self.save_images):
            depth_img = cv2.resize(depth, orig_img.shape[:2])
            cv2.imshow("depth", depth_img)
            self.gripper_cam_imgs.update(
                {"./%s_depth/img_%04d.jpg" % ("gripper", self.obs_it): depth_img * 255})
            im_dict = viz_aff_centers_preds(orig_img, aff_mask, aff_probs, center_dir,
                                            object_centers, object_masks,
                                            "gripper", self.obs_it,
                                            save_images=self.save_images)
            self.gripper_cam_imgs.update(im_dict)

        # Plot different objects
        cluster_outputs = []
        object_centers = [torch_to_numpy(o) for o in object_centers]
        if(len(object_centers) > 0):
            target_px = object_centers[0]
        else:
            return

        # To numpy
        aff_probs = torch_to_numpy(aff_probs)
        aff_probs = np.transpose(aff_probs[0], (1, 2, 0))  # H, W, 2
        object_masks = torch_to_numpy(object_masks[0])  # H, W

        max_robustness = 0
        obj_class = np.unique(object_masks)[1:]
        obj_class = obj_class[obj_class != 0]  # remove background class

        # Look for most likely center
        n_pixels = aff_mask.shape[1] * aff_mask.shape[2]
        pred_shape = aff_probs.shape[:2]
        orig_shape = depth.shape[:2]
        for i, o in enumerate(object_centers):
            # Mean prob of being class 1 (foreground)
            cluster = aff_probs[object_masks == obj_class[i], 1]
            robustness = np.mean(cluster)
            pixel_count = cluster.shape[0] / n_pixels

            # Convert back to observation size
            o = (o * orig_shape / pred_shape).astype("int64")
            v, u = o

            world_pt = pixel2world(cam, u, v, depth)
            c_out = {"center": world_pt,
                     "pixel_count": pixel_count,
                     "robustness": max_robustness}
            cluster_outputs.append(c_out)
            if(robustness > max_robustness):
                max_robustness = robustness
                target_px = o
                target_world = world_pt

        # p.removeAllUserDebugItems()

        # self.env.unwrapped.current_target = target_world
        # Maximum distance given the task
        for out_dict in cluster_outputs:
            c = out_dict["center"]
            # If aff detects closer target which is large enough
            # and Detected affordance close to target
            if(np.linalg.norm(self.env.unwrapped.current_target - c) < 0.05):
                self.env.unwrapped.current_target = c

        # See selected point
        # p.addUserDebugText("target",
        #                    textPosition=self.env.unwrapped.current_target,
        #                    textColorRGB=[1, 0, 0])