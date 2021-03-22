import gym
import cv2
import numpy as np
import torch
from torchvision import transforms
import hydra


class EnvWrapper(gym.ObservationWrapper):
    def __init__(self, env, history_length, skip_frames, img_size, use_img=False,
                 use_static_cam=False, use_depth=False, use_gripper_cam=False,
                 use_pos=False, transforms=None, train=False):
        super(EnvWrapper, self).__init__(env)
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
        self.static_id = 0
        self.gripper_id = 1
        self.find_cam_ids()
        # parameters to define observation
        self._use_img_obs = use_static_cam
        self._use_robot_obs = use_pos
        self._use_depth = use_depth
        self._use_gripper_img = use_gripper_cam
        self.observation_space = self.get_obs_space()
        self._training = train

    def find_cam_ids(self):
        for i, cam in enumerate(self.cameras):
            if "gripper" in cam.name:
                self.gripper_id = i
            else:
                self.static_id = i

    def get_obs_space(self):
        if(self._use_img_obs):
            obs_space_dict = {
                'img_obs': gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(self.history_length, self.img_size, self.img_size))
                    }
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

    def observation(self, obs):
        # "rgb_obs", "depth_obs", "robot_obs","scene_obs"
        if(self._use_img_obs or self._use_gripper_img):
            obs = {}
            obs_dict = self.get_obs()
            if(self._use_img_obs):
                # cv2.imshow("static_cam orig", obs_dict['rgb_obs'][self.static_id])
                img_obs = self.img_preprocessing(
                            obs_dict['rgb_obs'][self.static_id])
                obs["img_obs"] = img_obs
            if(self._use_gripper_img):
                # cv2.imshow("gripper_cam orig", obs_dict['rgb_obs'][self.gripper_id])
                gripper_obs = self.img_preprocessing(
                                obs_dict['rgb_obs'][self.gripper_id])
                obs["gripper_img_obs"] = gripper_obs
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
