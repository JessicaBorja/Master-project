import os
import logging
import torch
import numpy as np
import cv2
from scipy.spatial.transform.rotation import Rotation as R

import gym
from gym import spaces

from utils.img_utils import torch_to_numpy, viz_aff_centers_preds, visualize, get_depth_around_point
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
    def __init__(self, env, max_ts, img_size,
                 gripper_cam,
                 static_cam,
                 transforms=None,
                 use_pos=False,
                 affordance_cfg=None,
                 train=False,
                 save_images=False,
                 viz=False):

        super(RLWrapper, self).__init__(env)
        # REWARD FUNCTION
        self.affordance_cfg = affordance_cfg
        self.ts_counter = 0
        self.max_ts = max_ts
        if self.affordance_cfg.gripper_cam.densify_reward:
            print("RewardWrapper: Gripper cam to shape reward")

        # OBSERVATION
        self.img_size = img_size

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
        if(save_images):
            create_dirs = ['./gripper_depth/', "gripper_orig", "gripper_masks", "gripper_aff", "gripper_dirs"]
            for directory in create_dirs:
                os.makedirs(directory, exist_ok=True)
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
        # 0-2 -> position , 3 -> yaw angle 4 gripper action
        _action_space = np.ones(4)
        self.action_space = spaces.Box(_action_space * -1, _action_space)
        self.observation_space = get_obs_space(affordance_cfg,
                                               self.gripper_cam_cfg,
                                               self.static_cam_cfg,
                                               self.channels, self.img_size,
                                               self.use_robot_obs)

        # Save images
        self.gripper_cam_imgs = {}
        self.curr_detected_obj = None

        self.T_tcp_cam = self.env.env.camera_manager.gripper_cam.get_extrinsic_calibration('panda')

    def reset(self, *args, **kwargs):
        observation = self.env.reset(*args, **kwargs)
        return self.observation(observation)

    def step(self, action):
        action = np.append(action, 1)
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation), self.reward(reward, observation, done), done, info

    def observation(self, obs):
        # "rgb_obs", "depth_obs", "robot_obs","scene_obs"
        gripper_action = int(obs["robot_obs"][-1] > 0.4)
        new_obs = {**self.get_cam_obs(obs, "gripper",
                                      self.gripper_cam_aff_net,
                                      self.gripper_cam_cfg,
                                      self.affordance_cfg.gripper_cam),
                   **self.get_cam_obs(obs, "static",
                                      self.static_cam_aff_net,
                                      self.static_cam_cfg,
                                      self.affordance_cfg.static_cam),
                   "robot_obs": np.array([*obs["robot_obs"], gripper_action])
                   }
        self.obs_it += 1
        return new_obs

    def reward(self, rew, obs, done):
        # modify rew
        if self.affordance_cfg.gripper_cam.densify_reward:
            # set by observation wrapper so that
            # both have the same observation on
            # a given timestep

            tcp_pos = obs["robot_obs"][:3]

            if not done and self.ts_counter < self.max_ts - 1:
                distance = np.linalg.norm(tcp_pos - self.curr_detected_obj)
                # cannot be larger than 1
                # scale dist increases as it falls away from object
                scale_dist = min(distance / self.env.termination_radius, 1)
                rew += (1 - scale_dist)**0.5
            #     self.ts_counter += 1
            # else:
            #     # If episode was successful
            #     if rew >= 1:
            #         # Reward for remaining ts
            #         rew += self.max_ts - 1 - self.ts_counter
            #     self.ts_counter = 0
        return rew

    def get_cam_obs(self, obs_dict, cam_type, aff_net,
                    obs_cfg, aff_cfg):
        obs = {}
        if obs_cfg.use_depth:
            # Resize
            depth_obs = depth_preprocessing(
                            obs_dict['depth_%s' % cam_type],
                            self.img_size)
            obs["%s_depth_obs" % cam_type] = depth_obs
        if obs_cfg.use_img:
            # Transform rgb to grayscale
            img_obs = img_preprocessing(
                                obs_dict['rgb_%s' % cam_type],
                                self.rl_transforms)
            # 1, W, H
            obs["%s_img_obs" % cam_type] = img_obs

        get_gripper_target = cam_type == "gripper" and (
                    self.affordance_cfg.gripper_cam.densify_reward
                    or self.affordance_cfg.gripper_cam.target_in_obs
                    or self.affordance_cfg.gripper_cam.use_distance)

        if aff_net is not None and (aff_cfg.use or get_gripper_target):
            # Np array 1, H, W
            processed_obs = img_preprocessing(
                                obs_dict['rgb_%s' % cam_type],
                                self.aff_transforms[cam_type])
            with torch.no_grad():
                # 1, 1, H, W in range [-1, 1]
                obs_t = torch.tensor(processed_obs).unsqueeze(0)
                obs_t = obs_t.float().cuda()

                # 1, H, W
                # aff_logits, aff_probs, aff_mask, directions
                _, aff_probs, aff_mask, directions = aff_net(obs_t)
                # aff_mask = self._mask_transforms(aff_mask).cuda()
                if self.viz and cam_type == "static":
                    visualize(aff_mask,
                              obs_dict['rgb_%s' % cam_type][:, :, ::-1],
                              imshow=True)
                mask = torch_to_numpy(aff_mask)  # foreground/affordance Mask       
                if obs_cfg.use_img and aff_mask.shape[1:] != img_obs.shape[1:]:
                    new_shape = (aff_mask.shape[0], *img_obs.shape[1:])
                    mask = np.resize(mask, new_shape)
                if get_gripper_target:
                    preds = {"%s_aff" % cam_type: aff_mask,
                             "%s_center_dir" % cam_type: directions,
                             "%s_aff_probs" % cam_type: aff_probs}

                    # Computes newest target
                    gripper_cam = self.env.camera_manager.gripper_cam
                    self.find_target_center(gripper_cam,
                                            obs_dict['rgb_gripper'],
                                            obs_dict['depth_gripper'],
                                            preds)
            if(self.affordance_cfg.gripper_cam.target_in_obs):
                obs["detected_target_pos"] = self.curr_detected_obj
            if(self.affordance_cfg.gripper_cam.use_distance):
                distance = np.linalg.norm(self.curr_detected_obj
                                          - obs_dict["robot_obs"][:3]) if self.curr_detected_obj is not None else self.env.termination_radius
                obs["target_distance"] = np.array([distance])
            if(aff_cfg.use):
                # m = np.transpose(mask * 255,(1, 2, 0)).astype('uint8')
                # cv2.imshow("%s_aff" % cam_type, m)
                obs["%s_aff" % cam_type] = mask
        return obs

    def np_quat_to_scipy_quat(self, quat):
        """wxyz to xyzw"""
        return np.array([quat.x, quat.y, quat.z, quat.w])

    def pos_orn_to_matrix(self, pos, orn):
        """
        :param pos: np.array of shape (3,)
        :param orn: np.array of shape (4,) -> quaternion xyzw
                    np.quaternion -> quaternion wxyz
                    np.array of shape (3,) -> euler angles xyz
        :return: 4x4 homogeneous transformation
        """
        mat = np.eye(4)
        if isinstance(orn, np.quaternion):
            orn = self.np_quat_to_scipy_quat(orn)
            mat[:3, :3] = R.from_quat(orn).as_matrix()
        elif len(orn) == 4:
            mat[:3, :3] = R.from_quat(orn).as_matrix()
        elif len(orn) == 3:
            mat[:3, :3] = R.from_euler('xyz', orn).as_matrix()
        mat[:3, 3] = pos
        return mat

    # Aff-center
    def find_target_center(self, cam, orig_img, depth, obs):
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

        # Predict affordances and centers
        aff_mask, center_dir, object_centers, object_masks = \
            self.gripper_cam_aff_net.predict(aff_mask, directions)

        # Visualize predictions
        depth_img = cv2.resize(depth, orig_img.shape[:2])
        cv2.imshow("gripper_depth", depth_img)
        im_dict = viz_aff_centers_preds(orig_img, aff_mask, aff_probs, center_dir,
                                        object_centers, object_masks,
                                        "gripper", self.obs_it,
                                        save_images=self.save_images)
        # im_dict.update({"./gripper_depth/img_%04d.png" % self.obs_it:
        #                 (255 * (depth_img - depth_img.min()) / (depth_img.max() - depth_img.min())).astype(np.uint8)})
        # for im_name, im in im_dict.items():
        #     cv2.imwrite(im_name, im)
        # self.gripper_cam_imgs.update(im_dict)

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

            o = o[::-1]

            # Convert back to observation size
            o = (o * orig_shape / pred_shape).astype("int64")
            o, depth_non_zero = get_depth_around_point(o, depth)
            if depth_non_zero:
                cam_frame_pt = cam.deproject(o, depth)
                tcp_pos, tcp_orn = self.env.robot.get_tcp_pos_orn()
                tcp_mat = self.pos_orn_to_matrix(tcp_pos, tcp_orn)
                world_pt = tcp_mat @ self.T_tcp_cam @ np.array([*cam_frame_pt, 1])
                world_pt = world_pt[:3]
                c_out = {"center": world_pt,
                         "pixel_count": pixel_count,
                         "robustness": robustness}
                cluster_outputs.append(c_out)

        # Maximum distance given the task
        most_robust = 0
        if(self.curr_detected_obj is not None):
            for out_dict in cluster_outputs:
                c = out_dict["center"]
                # If aff detects closer target which is large enough
                # and Detected affordance close to target
                dist = np.linalg.norm(self.curr_detected_obj - c)
                if(dist < self.env.termination_radius/2):
                    if(out_dict["robustness"] > most_robust):
                        self.curr_detected_obj = c
                        most_robust = out_dict["robustness"]
