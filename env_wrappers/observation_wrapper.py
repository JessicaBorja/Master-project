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

from sac_agent.sac_utils.utils import tt
from affordance_model.segmentator_centers import Segmentator
from utils.cam_projections import pixel2world
from utils.img_utils import torch_to_numpy, viz_aff_centers_preds


class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, history_length, skip_frames, img_size,
                 use_static_cam=False, use_depth=False, use_gripper_cam=False,
                 use_pos=False, transforms=None, train=False, affordance=None):
        super(ObservationWrapper, self).__init__(env)
        self.env = env
        self.img_size = img_size
        shape = (1, img_size, img_size)
        if(transforms):
            self._transforms_cfg =\
                transforms["train"] if train else transforms["validation"]
            self.transforms, shape = self.get_transforms()

        # History length
        self.skip_frames = skip_frames
        self.history_length = history_length * shape[0]
        self.obs_count = 0
        _n_images = ((skip_frames+1)*(history_length-1) + 1)
        self._total_frames = np.zeros((_n_images * shape[0],
                                       self.img_size, self.img_size))
        self._indices = [i for i in range(_n_images)
                         if i % (skip_frames+1) == 0]
        assert len(self._indices) == history_length
        self._cur_img_obs = None
        # Cameras defaults
        self.static_id, self.gripper_id = self.find_cam_ids()
        # Parameters to define observation
        self.affordance = affordance
        self._use_img_obs = use_static_cam
        self._use_robot_obs = use_pos
        self._use_depth = use_depth
        self._use_gripper_img = use_gripper_cam
        self.observation_space = self.get_obs_space()
        self._training = train
        # Parameters to store affordance
        self.gripper_cam_aff_net = self.init_aff_net('gripper')
        self.static_cam_aff_net = self.init_aff_net('static')
        self.curr_raw_obs = None
        self.curr_processed_obs = None
        # 0-2 -> position , 3 -> yaw angle, 4 gripper action
        _action_space = np.ones(5)
        self.action_space = spaces.Box(_action_space * -1, _action_space)

    def find_cam_ids(self):
        static_id, gripper_id = 0, 1
        for i, cam in enumerate(self.cameras):
            if "gripper" in cam.name:
                gripper_id = i
            else:
                static_id = i
        return static_id, gripper_id

    def get_obs_space(self):
        if(self._use_img_obs or self._use_gripper_img):
            obs_space_dict = {}
            if(self._use_img_obs):
                obs_space_dict["img_obs"] = gym.spaces.Box(
                    low=0, high=255,
                    shape=(self.history_length, self.img_size, self.img_size))
            if(self._use_depth):
                obs_space_dict['depth_obs'] = gym.spaces.Box(
                    low=0, high=255,
                    shape=(self.history_length, self.img_size, self.img_size))
            if(self._use_robot_obs):
                # *tcp_pos(3), *tcp_euler(3),gripper_opening_width(1),
                obs_space_dict['robot_obs'] = gym.spaces.Box(
                    low=-0.5, high=0.5, shape=(7,))
            if(self._use_gripper_img):
                obs_space_dict['gripper_img_obs'] = gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(self.history_length, self.img_size, self.img_size))
            if(self.affordance.gripper_cam.target_in_obs):
                obs_space_dict['detected_target_pos'] = gym.spaces.Box(
                    low=-1, high=1, shape=(3,))
            return gym.spaces.Dict(obs_space_dict)
        else:
            return gym.spaces.Box(low=-0.5, high=0.5, shape=(10,))

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
                    print("obs_wrapper: Path does not exist: %s" % path)
            elif(self.affordance.gripper_cam.use
                 and cam_str == "gripper"):
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
                    print("obs_wrapper: Path does not exist: %s" % path)
        return aff_net

    def get_static_obs(self, obs_dict):
        obs = {}
        if(self._use_img_obs):
            # cv2.imshow("static_cam orig",
            #            obs_dict['rgb_obs'][self.static_id])
            # cv2.waitKey(1)
            img_obs = self.img_preprocessing(
                        obs_dict['rgb_obs'][self.static_id])
            obs["img_obs"] = img_obs
        if(self.static_cam_aff_net is not None
           and self.affordance.static_cam.use):
            img_obs = self.img_preprocessing(
                        obs_dict['rgb_obs'][self.static_id])
            with torch.no_grad():
                # 1, 1, W, H
                obs_t = torch.tensor(img_obs).unsqueeze(0)
                obs_t = obs_t.float().cuda()
                _, _, mask, _ = self.static_cam_aff_net(obs_t)
                mask = mask[0].cpu().detach().numpy()
                del obs_t
            del img_obs
            obs["static_aff"] = mask
        return obs

    # Segmentator centers model
    def get_gripper_obs(self, obs_dict):
        obs = {}
        if(self._use_gripper_img):
            # cv2.imshow("gripper_cam orig",
            #            obs_dict['rgb_obs'][self.gripper_id])
            gripper_obs = self.img_preprocessing(
                            obs_dict['rgb_obs'][self.gripper_id])
            # 1, W, H
            obs["gripper_img_obs"] = gripper_obs
        if(self.gripper_cam_aff_net is not None
           and (self.affordance.gripper_cam.use or
                self.affordance.gripper_cam.densify_reward)):
            # Np array 1, H, W
            gripper_obs = self.img_preprocessing(
                        obs_dict['rgb_obs'][self.gripper_id])
            with torch.no_grad():
                # 1, 1, H, W in range [-1, 1]
                obs_t = torch.tensor(gripper_obs).unsqueeze(0)
                obs_t = obs_t.float().cuda()

                # 1, H, W
                # aff_logits, aff_probs, aff_mask, directions
                _, aff_probs, aff_mask, directions = \
                    self.gripper_cam_aff_net(obs_t)
                mask = torch_to_numpy(aff_mask)  # foreground/affordance Mask
                preds = {"gripper_aff": mask,
                         "gripper_center_dir": torch_to_numpy(directions),
                         "gripper_aff_probs": torch_to_numpy(aff_probs)}
                # Computes newest target
                self.find_target_center(self.gripper_id,
                                        obs_dict['rgb_obs'][self.gripper_id],
                                        obs_dict['depth_obs'][self.gripper_id],
                                        preds)
                del obs_t
            obs["gripper_aff"] = mask
            obs["detected_target_pos"] = self.unwrapped.current_target
            del gripper_obs
        return obs

    def observation(self, obs):
        # "rgb_obs", "depth_obs", "robot_obs","scene_obs"
        if(self._use_img_obs or self._use_gripper_img):
            obs = {}
            self.curr_raw_obs = self.get_obs()
            obs_dict = self.curr_raw_obs
            obs = {**self.get_gripper_obs(obs_dict),
                   **self.get_static_obs(obs_dict)}
            if(self._use_depth):
                depth_obs = self.depth_preprocessing(
                                obs_dict['depth_obs'][self.static_id])
                obs["depth_obs"] = depth_obs
            if(self._use_robot_obs):
                # *tcp_pos(3), *tcp_euler(3), gripper_opening_width(1),
                obs["robot_obs"] = obs_dict["robot_obs"][:7]
            self.obs_count += 1
        else:
            robot_obs, scene_obs = obs['robot_obs'], obs['scene_obs']
            obs = np.concatenate((robot_obs[:7],  # only pos and euler orn
                                  scene_obs[:3]))  # only doors states
        self.curr_processed_obs = obs
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

    def get_transforms(self):
        transforms_lst = []
        for cfg in self._transforms_cfg:
            transforms_lst.append(hydra.utils.instantiate(cfg))
        apply_transforms = transforms.Compose(transforms_lst)
        test_tensor = torch.zeros((3, self.img_size, self.img_size))
        test_tensor = apply_transforms(test_tensor)
        # History length
        if(test_tensor.shape == 2):
            test_tensor = test_tensor.unsqueeze(0)
        return apply_transforms, test_tensor.shape

    def img_preprocessing(self, frame):
        # obs is from 0-255, (img_size, img_size, 3)
        frame = torch.from_numpy(frame).permute(2, 0, 1)
        frame = self.transforms(frame)
        frame = frame.cpu().detach().numpy()

        # History length
        if(frame.shape == 2):
            frame = np.expand_dims(frame, 0)  # (1, img_size, img_size)
        c, w, h = frame.shape
        self._total_frames = np.pad(self._total_frames, ((
            c, 0), (0, 0), (0, 0)), mode='constant')[:-c, :, :]
        self._total_frames[0:c] = frame

        self._cur_img_obs = [self._total_frames[i * c:(i * c) + c]
                             for i in self._indices]
        self._cur_img_obs = np.concatenate(self._cur_img_obs, 0)
        return self._cur_img_obs

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
            self.gripper_cam_aff_net.predict(tt(aff_mask), tt(directions))

        # Visualize predictions
        viz_aff_centers_preds(orig_img, aff_mask, tt(aff_probs), center_dir,
                              object_centers, object_masks)

        # Plot different objects
        cluster_outputs = []
        object_centers = [torch_to_numpy(o) for o in object_centers]
        if(len(object_centers) > 0):
            target_px = object_centers[0]
        else:
            return

        # To numpy
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
        v, u = target_px
        out_img = cv2.drawMarker(np.array(orig_img[:, :, ::-1]),
                                 (u, v),
                                 (0, 255, 0),
                                 markerType=cv2.MARKER_CROSS,
                                 markerSize=12,
                                 line_type=cv2.LINE_AA)
        depth = cv2.drawMarker(np.array(depth),
                               (u, v),
                               (0, 255, 0),
                               markerType=cv2.MARKER_CROSS,
                               markerSize=12,
                               line_type=cv2.LINE_AA)
        cv2.imshow("out_img", out_img)
        cv2.imshow("depth", depth)

        p.removeAllUserDebugItems()
        self.unwrapped.current_target = target_world
        # Maximum distance given the task
        # for out_dict in cluster_outputs:
        #     c = out_dict["center"]
        #     # If aff detects closer target which is large enough
        #     # and Detected affordance close to target
        #     if(np.linalg.norm(self.unwrapped.current_target - c) < 0.05):
        #         self.unwrapped.current_target = c

        # See selected point
        p.addUserDebugText("target",
                           textPosition=self.unwrapped.current_target,
                           textColorRGB=[1, 0, 0])
