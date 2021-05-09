import gym
import cv2
import numpy as np
import torch
from torchvision import transforms
from affordance_model.segmentator import Segmentator
from sac_agent.sac_utils.utils import show_mask_np
import os
import hydra


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
        self._use_img_obs = use_static_cam
        self._use_robot_obs = use_pos
        self._use_depth = use_depth
        self._use_gripper_img = use_gripper_cam
        self.observation_space = self.get_obs_space()
        self._training = train
        # Parameters to store affordance
        self.affordance = affordance
        self.gripper_cam_aff_net = self.init_aff_net('gripper')
        self.static_cam_aff_net = self.init_aff_net('static')
        self.curr_raw_obs = None
        self.curr_processed_obs = None

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
                    print("ENV: Static cam affordance model loaded")
                else:
                    self.affordance = None
                    path = os.path.abspath(path)
                    print("Path does not exist: %s" % path)
            elif(self.affordance.gripper_cam.use
                 and cam_str == "gripper"):
                path = self.affordance.gripper_cam.model_path
                if(os.path.exists(path)):
                    aff_net = Segmentator.load_from_checkpoint(
                                        path,
                                        cfg=self.affordance.hyperparameters)
                    aff_net.cuda()
                    aff_net.eval()
                    print("ENV: gripper affordance model loaded")
                else:
                    self.affordance = None
                    path = os.path.abspath(path)
                    print("Path does not exist: %s" % path)
        return aff_net

    def get_static_obs(self, obs_dict):
        obs = {}
        if(self._use_img_obs):
            # cv2.imshow("static_cam orig",
            #            obs_dict['rgb_obs'][self.static_id])
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
                mask = self.static_cam_aff_net(obs_t)
                mask = torch.argmax(mask, axis=1, keepdim=True)
                mask = mask[0].cpu().detach().numpy()
                del obs_t
            del img_obs
            obs["static_aff"] = mask
        return obs

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
           and self.affordance.gripper_cam.use):
            # Np array 1, H, W
            gripper_obs = self.img_preprocessing(
                        obs_dict['rgb_obs'][self.gripper_id])
            with torch.no_grad():
                # 1, 1, H, W in range [-1, 1]
                obs_t = torch.tensor(gripper_obs).unsqueeze(0)
                obs_t = obs_t.float().cuda()

                # 1, 2, H, W in [0-1]
                # softmax output
                # mask = self.gripper_cam_aff_net.predict(obs_t)
                # mask = mask[0, 1].cpu().detach().numpy()
                # mask = np.expand_dims(mask, 0)

                # 1, H, W
                # Tresholded mask ..
                mask = self.gripper_cam_aff_net(obs_t)
                mask = torch.argmax(mask, axis=1, keepdim=True)
                mask = mask[0].cpu().detach().numpy()
                # show_mask_np(gripper_obs, mask)
                del obs_t
            obs["gripper_aff"] = mask
            del gripper_obs
        return obs

    def observation(self, obs):
        # "rgb_obs", "depth_obs", "robot_obs","scene_obs"
        if(self._use_img_obs or self._use_gripper_img):
            obs = {}
            self.curr_raw_obs = self.get_obs()
            obs_dict = self.curr_raw_obs
            if(self._use_gripper_img or self._use_img_obs):
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
