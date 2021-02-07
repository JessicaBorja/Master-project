import gym
import torch
import cv2
import numpy as np

class EnvWrapper(gym.ObservationWrapper):
    def __init__(self, env, use_img, use_depth, use_pos, history_length, skip_frames, img_size, train=False):
        super(EnvWrapper, self).__init__(env)
        self.env = env

        #settup to stack images
        self.skip_frames = skip_frames
        self.history_length = history_length
        self.obs_count = 0
        self.img_size = img_size
        self._total_frames = np.zeros(( (skip_frames+1)*(self.history_length-1) + 1, self.img_size,self.img_size))
        self._indices = [i for i in range(self._total_frames.shape[0]) if i%(skip_frames+1)==0]
        assert len(self._indices) == history_length
        self._cur_img_obs = None

        #parameters to define observation
        self._use_img_obs = use_img
        self._use_robot_obs = use_pos
        self._use_depth = use_depth
        self.observation_space = self.get_obs_space(use_img, use_depth, use_pos)
        self._training = train

    def get_obs_space(self, use_img, use_depth, use_pos):
        if(use_img):
            obs_space_dict = {'img_obs': gym.spaces.Box(low=0, high=255, shape=(self.history_length, self.img_size, self.img_size)),}
            if(use_depth):
                obs_space_dict['depth_obs'] = gym.spaces.Box(low=0, high=255, shape=(self.history_length, self.img_size, self.img_size))
            if(use_pos):
                # *tcp_pos(3), *tcp_euler(3),gripper_opening_width(1),
                obs_space_dict['robot_obs'] = gym.spaces.Box(low=-0.5, high=0.5, shape=(7,))
            return gym.spaces.Dict(obs_space_dict)
        else:
            # robot_obs= 3 for pos, 3 for angle and 1 for gripper. idk how to define the obs space for this.
            # obs_space_dict = {
            #     "scene_obs":  gym.spaces.Box(low=0, high=1.5, shape=(3,)), #each door has different ranges
            #     'robot_obs': gym.spaces.Box(low=-0.5, high=0.5, shape=(7,))
            # }
            return gym.spaces.Box(low=-0.5, high=0.5, shape=(10,))

    def observation(self, obs):
        if(self._use_img_obs):
            obs_dict = self.get_obs()# "rgb_obs", "depth_obs", "robot_obs","scene_obs"
            img_obs = self.img_preprocessing(obs_dict['rgb_obs'][0][:, :, ::-1]) #from rgb to grayscale
            obs = {"img_obs": img_obs}
            if(self._use_depth):
                depth_obs = self.depth_preprocessing(obs_dict['depth_obs'][0])
                obs["depth_obs"] = depth_obs
            if(self._use_robot_obs): # *tcp_pos(3), *tcp_euler(3),gripper_opening_width(1),
                obs["robot_obs"] = obs_dict["robot_obs"][:7]
            self.obs_count +=1
        else:
            robot_obs, scene_obs =  obs['robot_obs'], obs['scene_obs']
            obs = np.concatenate((robot_obs[:7], #only pos and euler orn
                                  scene_obs[:3])) #only doors states
        return obs
        
    def depth_preprocessing(self, frame): #obs is from 0-255, (300,300,1)
        new_frame = cv2.resize(
            frame, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA) #(img_size, img_size)
        new_frame =  np.expand_dims(new_frame, axis=0) #(1, img_size, img_size)
        return new_frame

    def img_preprocessing(self, frame): #obs is from 0-255, (300,300,3)
        new_frame = cv2.resize(
            frame, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA) #(img_size, img_size)
        new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY).astype(float)

        #"Transforms"
        #norm_image = np.copy(new_frame)
        cv2.normalize(new_frame, new_frame, -1, 1, norm_type = cv2.NORM_MINMAX)#(0,255) -> (-1,1)
        if(self._training): #add gaussian noise
            new_frame += np.random.normal(0, 0.01, size = new_frame.shape)

        #History length
        new_frame = np.expand_dims(new_frame, 0)#(1, img_size, img_size)
        self._total_frames = np.pad(self._total_frames,((1,0),(0,0),(0,0)), mode='constant')[:-1, :, :]
        self._total_frames[0] = new_frame
        
        self._cur_img_obs = self._total_frames[self._indices]
        return self._cur_img_obs