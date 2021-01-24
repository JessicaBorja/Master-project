import gym
import torch
import cv2
import numpy as np

class ImgWrapper(gym.ObservationWrapper):
    def __init__(self, env = None, skip_frames = 1, history_length = 3, img_size = 100):
        super(ImgWrapper, self).__init__(env)
        self.env = env
        spaces = {
            'position': gym.spaces.Box(low=-0.5, high=0.5, shape=(7,)),
            'rgb_obs': gym.spaces.Box(low=0, high=255, shape=(history_length,img_size,img_size)),
            'depth_obs': gym.spaces.Box(low=0, high=255, shape=(history_length,img_size,img_size))
        }
        self.observation_space = gym.spaces.Dict(spaces)
        self.skip_frames = skip_frames
        self.history_length = history_length
        self.obs_count = 0
        self.img_size = img_size
        self._total_frames = np.zeros(( (skip_frames+1)*(self.history_length-1) + 1, self.img_size,self.img_size))
        self._indices = [i for i in range(self._total_frames.shape[0]) if i%(skip_frames+1)==0]
        assert len(self._indices) == history_length
        self._img_obs = None

    def observation(self, obs):
        obs_dict = self.get_obs()# "rgb_obs", "depth_obs", "robot_obs","scene_obs"
        img_obs = self.img_preprocessing(obs_dict['rgb_obs'][0][:, :, ::-1])
        depth_obs = self.depth_preprocessing(obs_dict['depth_obs'][0])
        obs = {"rgb_obs": img_obs,
                "depth_obs": depth_obs,
                "position": obs_dict['robot_obs'][:7]
                }
        self.obs_count +=1
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
        
        self._img_obs = self._total_frames[self._indices]

        # if(self.obs_count == 0): #pos 0 is the most recent one
        #     zeros = np.zeros((self.history_length-1,self.img_size,self.img_size))
        #     self._img_obs = np.concatenate((new_frame, zeros), axis=0)
        # elif(self.obs_count%(self.skip_frames+1)==0):            
        #     self._img_obs = np.pad(self._img_obs,((1,0),(0,0),(0,0)), mode='constant')[:-1, :, :]
        #     self._img_obs[0] = new_frame
        return self._img_obs