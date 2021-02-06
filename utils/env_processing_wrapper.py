import gym
import torch
import cv2
import numpy as np

class EnvWrapper(gym.ObservationWrapper):
    def __init__(self, env = None, img_obs=False, img_processing={}):
        super(EnvWrapper, self).__init__(env)
        self.env = env
        skip_frames, history_length, img_size = img_processing['skip_frames'], img_processing['history_length'], img_processing['img_size']
        self.skip_frames = skip_frames
        self.history_length = history_length
        self.obs_count = 0
        self.img_size = img_size
        self._total_frames = np.zeros(( (skip_frames+1)*(self.history_length-1) + 1, self.img_size,self.img_size))
        self._indices = [i for i in range(self._total_frames.shape[0]) if i%(skip_frames+1)==0]
        assert len(self._indices) == history_length
        self._cur_img_obs = None
        self._use_img_obs = img_obs
        self.observation_space = self.get_obs_space(img_obs)


    def get_obs_space(self, img_obs):
        if(img_obs):
            # robot_obs= 3 for pos, 3 for angle and 1 for gripper. idk how to define the obs space for this.
            obs_space_dict = {
                'robot_obs': gym.spaces.Box(low=-0.5, high=0.5, shape=(7,)), 
                'img_obs': gym.spaces.Box(low=0, high=255, shape=(self.history_length, self.img_size, self.img_size)),
                'depth_obs': gym.spaces.Box(low=0, high=255, shape=(self.history_length, self.img_size, self.img_size))
            }
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
            depth_obs = self.depth_preprocessing(obs_dict['depth_obs'][0])
            obs = {"img_obs": img_obs,
                    "depth_obs": depth_obs,
                    "robot_obs": obs_dict['robot_obs'][:7]
                    }
            self.obs_count +=1
        else:
            robot_obs, scene_obs =  obs['robot_obs'], obs['scene_obs']
            obs = np.concatenate((robot_obs[:7], #only pos and euler orn
                                  scene_obs[:3])) #only doors states
        return obs
        
    def depth_preprocessing(self, frame):
        new_frame = cv2.resize(
            frame, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA) #(img_size, img_size)
        new_frame =  np.expand_dims(new_frame, axis=0) #(1, img_size, img_size)
        return new_frame

    def img_preprocessing(self, frame): #obs is from 0-255, (300,300,3)
        new_frame = cv2.resize(
            frame, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA) #(img_size, img_size)
        new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
        #cv2.normalize(new_frame, new_frame, 0, 255, norm_type = cv2.NORM_MINMAX)
        # cv2.imshow("win",new_frame)
        # cv2.waitKey(1)
        new_frame = np.expand_dims(new_frame, 0)#(1, img_size, img_size)
        self._total_frames = np.pad(self._total_frames,((1,0),(0,0),(0,0)), mode='constant')[:-1, :, :]
        self._total_frames[0] = new_frame
        
        self._cur_img_obs = self._total_frames[self._indices]

        # if(self.obs_count == 0): #pos 0 is the most recent one
        #     zeros = np.zeros((self.history_length-1,self.img_size,self.img_size))
        #     self._img_obs = np.concatenate((new_frame, zeros), axis=0)
        # elif(self.obs_count%(self.skip_frames+1)==0):            
        #     self._img_obs = np.pad(self._img_obs,((1,0),(0,0),(0,0)), mode='constant')[:-1, :, :]
        #     self._img_obs[0] = new_frame
        return self._cur_img_obs