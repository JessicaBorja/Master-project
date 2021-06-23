import gym
import cv2
import numpy as np
from omegaconf.omegaconf import OmegaConf
import torch
from torchvision import transforms
import os
import hydra
from gym import spaces

import pybullet as p

from affordance_model.segmentator_centers import Segmentator
from affordance_model.utils.transforms import DistanceTransform
from utils.cam_projections import pixel2world
from utils.img_utils import torch_to_numpy, viz_aff_centers_preds


class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, history_length, skip_frames, img_size,
                 gripper_cam, static_cam, use_pos=False, affordance=None,
                 transforms=None, train=False, save_images=False):
        super(ObservationWrapper, self).__init__(env)
        self.env = env
        self.img_size = img_size
        shape = (1, img_size, img_size)

        # Prepreocessing for affordance model
        _transforms_cfg =\
            affordance.transforms["train"] if train else affordance.transforms["validation"]
        self.aff_transforms, _ = self.get_transforms(_transforms_cfg)

        # Preprocessing for RL policy obs
        _transforms_cfg =\
            transforms["train"] if train else transforms["validation"]
        self.rl_transforms, shape = self.get_transforms(_transforms_cfg)
        self.channels = shape[0]

        # History length
        # self.skip_frames = skip_frames
        # self.history_length = history_length * shape[0]
        # _n_images = ((skip_frames+1)*(history_length-1) + 1)
        # self._total_frames = np.zeros((_n_images * shape[0],
        #                                self.img_size, self.img_size))
        # self._indices = [i for i in range(_n_images)
        #                  if i % (skip_frames+1) == 0]
        # assert len(self._indices) == history_length
        # self._cur_img_obs = None

        # Cameras defaults
        self.static_id, self.gripper_id = self.find_cam_ids()
        self.obs_it = 0
        self.save_images = save_images

        # Parameters to define observation
        self.affordance = affordance
        self.gripper_cam_cfg = gripper_cam
        self.static_cam_cfg = static_cam
        self._use_robot_obs = use_pos
        self.observation_space = self.get_obs_space()
        self._mask_transforms = DistanceTransform()

        # Parameters to store affordance
        self.gripper_cam_aff_net = self.init_aff_net('gripper')
        self.static_cam_aff_net = self.init_aff_net('static')
        self.curr_raw_obs = None

        if(self.unwrapped.task == "pickup"):
            # 0-2 -> position , 3 -> yaw angle 4 gripper action
            _action_space = np.ones(5)
        else:
            # 0-2 -> position , 3-5 -> orientation, 6 gripper action
            _action_space = np.ones(7)
        self.action_space = spaces.Box(_action_space * -1, _action_space)

        # Save images
        self.gripper_cam_imgs = {}

    def find_cam_ids(self):
        static_id, gripper_id = 0, 1
        for i, cam in enumerate(self.cameras):
            if "gripper" in cam.name:
                gripper_id = i
            elif "static" in cam.name:
                static_id = i
            elif "test" in cam.name:
                self.test_cam_id = i
        return static_id, gripper_id

    def get_obs_space(self):
        cfg_dict = {
            "gripper": [self.gripper_cam_cfg, self.affordance.gripper_cam],
            "static": [self.static_cam_cfg, self.affordance.static_cam]
        }
        obs_space_dict = {}
        for cam_type, config in cfg_dict.items():
            env_obs_cfg, aff_cfg = config
            if(env_obs_cfg.use_img):
                obs_space_dict["%s_img_obs" % cam_type] = gym.spaces.Box(
                    low=0, high=255,
                    shape=(self.channels, self.img_size, self.img_size))
            if(env_obs_cfg.use_depth):
                obs_space_dict['%s_depth_obs' % cam_type] = gym.spaces.Box(
                    low=0, high=255,
                    shape=(1, self.img_size, self.img_size))
            if(aff_cfg.use):
                obs_space_dict['%s_aff' % cam_type] = gym.spaces.Box(
                    low=0, high=1,
                    shape=(1, self.img_size, self.img_size))
        if(self._use_robot_obs):
            # *tcp_pos(3), *tcp_euler(1), gripper_width, gripper_action(1),
            if(self.unwrapped.task == "pickup"):
                obs_space_dict['robot_obs'] = gym.spaces.Box(
                    low=-0.5, high=0.5, shape=(6,))
            else:
                obs_space_dict['robot_obs'] = gym.spaces.Box(
                    low=-0.5, high=0.5, shape=(8,))
        if(self.affordance.gripper_cam.target_in_obs):
            obs_space_dict['detected_target_pos'] = gym.spaces.Box(
                low=-1, high=1, shape=(3,))
        if(self.affordance.gripper_cam.use_distance):
            obs_space_dict['target_distance'] = gym.spaces.Box(
                low=-1, high=1, shape=(1,))
        return gym.spaces.Dict(obs_space_dict)

    def init_aff_net(self, cam_str):
        aff_net = None
        if(self.affordance):
            if(self.affordance.static_cam.use
               and cam_str == "static"):
                path = self.affordance.static_cam.model_path
                if(os.path.exists(path)):
                    aff_net = Segmentator.load_from_checkpoint(
                                        path,
                                        cfg=self.affordance.hyperparameters)
                    aff_net.cuda()
                    aff_net.eval()
                    print("obs_wrapper: Static cam affordance model loaded")
                else:
                    self.affordance = None
                    path = os.path.abspath(path)
                    raise TypeError("Path does not exist: %s" % path)
            elif(cam_str == "gripper" and
                 (self.affordance.gripper_cam.use or
                  self.affordance.gripper_cam.densify_reward)):
                path = self.affordance.gripper_cam.model_path

                # Configuration of the model
                hp = {**self.affordance.hyperparameters,
                      "hough_voting": self.affordance.gripper_cam.hough_voting}
                hp = OmegaConf.create(hp)

                # Create model
                if(os.path.exists(path)):
                    aff_net = Segmentator.load_from_checkpoint(path, cfg=hp)
                    aff_net.cuda()
                    aff_net.eval()
                    print("obs_wrapper: gripper affordance model loaded")
                else:
                    self.affordance = None
                    path = os.path.abspath(path)
                    raise TypeError("Path does not exist: %s" % path)
        return aff_net

    def get_cam_obs(self, obs_dict, cam_type, aff_net,
                    obs_cfg, aff_cfg, cam_id):
        obs = {}
        if(obs_cfg.use_depth):
            # Resize
            depth_obs = self.depth_preprocessing(
                            obs_dict['depth_obs'][cam_id])
            obs["%s_depth_obs" % cam_type] = depth_obs
        if(obs_cfg.use_img):
            # Transform rgb to grayscale
            img_obs = self.img_preprocessing(
                                obs_dict['rgb_obs'][cam_id],
                                self.rl_transforms)
            # 1, W, H
            obs["%s_img_obs" % cam_type] = img_obs

        get_gripper_target = cam_type == "gripper" and (
                    self.affordance.gripper_cam.densify_reward
                    or self.affordance.gripper_cam.target_in_obs
                    or self.affordance.gripper_cam.use_distance)

        if(aff_net is not None and (aff_cfg.use or get_gripper_target)):
            # Np array 1, H, W
            grayscale_obs = self.img_preprocessing(
                        obs_dict['rgb_obs'][cam_id],
                        self.aff_transforms)
            with torch.no_grad():
                # 1, 1, H, W in range [-1, 1]
                obs_t = torch.tensor(grayscale_obs).unsqueeze(0)
                obs_t = obs_t.float().cuda()

                # 1, H, W
                # aff_logits, aff_probs, aff_mask, directions
                _, aff_probs, aff_mask, directions = aff_net(obs_t)
                # aff_mask = self._mask_transforms(aff_mask).cuda()
                mask = torch_to_numpy(aff_mask)  # foreground/affordance Mask                
                if(get_gripper_target):
                    preds = {"%s_aff" % cam_type: aff_mask,
                             "%s_center_dir" % cam_type: directions,
                             "%s_aff_probs" % cam_type: aff_probs}

                    # Computes newest target
                    self.find_target_center(self.gripper_id,
                                            obs_dict['rgb_obs'][self.gripper_id],
                                            obs_dict['depth_obs'][self.gripper_id],
                                            preds)
            if(self.affordance.gripper_cam.target_in_obs):
                obs["detected_target_pos"] = self.unwrapped.current_target
            if(self.affordance.gripper_cam.use_distance):
                distance = np.linalg.norm(self.unwrapped.current_target
                                          - obs_dict["robot_obs"][:3])
                obs["target_distance"] = np.array([distance])
            if(aff_cfg.use):
                # m = np.transpose(mask * 255,(1, 2, 0)).astype('uint8')
                # cv2.imshow("%s_aff" % cam_type, m)
                obs["%s_aff" % cam_type] = mask
        return obs

    def observation(self, obs):
        # "rgb_obs", "depth_obs", "robot_obs","scene_obs"
        obs = {}
        self.curr_raw_obs = self.get_obs()
        obs_dict = self.curr_raw_obs
        obs = {**self.get_cam_obs(obs_dict, "gripper",
                                  self.gripper_cam_aff_net,
                                  self.gripper_cam_cfg,
                                  self.affordance.gripper_cam,
                                  self.gripper_id),
               **self.get_cam_obs(obs_dict, "static",
                                  self.static_cam_aff_net,
                                  self.static_cam_cfg,
                                  self.affordance.static_cam,
                                  self.static_id)}
        if(self._use_robot_obs):
            if(self.unwrapped.task == "pickup"):
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
        self.curr_processed_obs = obs
        self.obs_it += 1
        return obs

    def depth_preprocessing(self, frame):
        # obs is from 0-255, (img_size, img_size, 1)
        new_frame = cv2.resize(
                frame,
                (self.img_size, self.img_size),
                interpolation=cv2.INTER_AREA)
        # (1, img_size, img_size)
        new_frame = np.expand_dims(new_frame, axis=0)
        return new_frame

    def get_transforms(self, transforms_cfg):
        transforms_lst = []
        for cfg in transforms_cfg:
            transforms_lst.append(hydra.utils.instantiate(cfg))
        apply_transforms = transforms.Compose(transforms_lst)
        test_tensor = torch.zeros((3, self.img_size, self.img_size))
        test_tensor = apply_transforms(test_tensor)
        # History length
        if(test_tensor.shape == 2):
            test_tensor = test_tensor.unsqueeze(0)
        return apply_transforms, test_tensor.shape

    def img_preprocessing(self, frame, transforms):
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

    # Aff-center
    def find_target_center(self, cam_id, orig_img, depth, obs):
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
        # im_dict = viz_aff_centers_preds(orig_img, aff_mask, aff_probs, center_dir,
        #                                 object_centers, object_masks,
        #                                 "gripper", self.obs_it,
        #                                 save_images=self.save_images)
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

        # world cord
        # v, u = target_px
        # out_img = cv2.drawMarker(np.array(orig_img[:, :, ::-1]),
        #                          (u, v),
        #                          (0, 255, 0),
        #                          markerType=cv2.MARKER_CROSS,
        #                          markerSize=12,
        #                          line_type=cv2.LINE_AA)
        # depth = cv2.drawMarker(np.array(depth),
        #                        (u, v),
        #                        (0, 255, 0),
        #                        markerType=cv2.MARKER_CROSS,
        #                        markerSize=12,
        #                        line_type=cv2.LINE_AA)
        # cv2.imshow("out_img", out_img)
        # cv2.imshow("depth", depth)

        # p.removeAllUserDebugItems()

        # self.unwrapped.current_target = target_world
        # Maximum distance given the task
        for out_dict in cluster_outputs:
            c = out_dict["center"]
            # If aff detects closer target which is large enough
            # and Detected affordance close to target
            if(np.linalg.norm(self.unwrapped.current_target - c) < 0.05):
                self.unwrapped.current_target = c

        # See selected point
        # p.addUserDebugText("target",
        #                    textPosition=self.unwrapped.current_target,
        #                    textColorRGB=[1, 0, 0])
