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
            'rgb_obs': gym.spaces.Box(low=0, high=255, shape=(history_length,100,100))
        }
        self.observation_space = gym.spaces.Dict(spaces)
        self.skip_frames = skip_frames
        self.history_length = history_length
        self.obs_count = 0
        self.img_size = img_size
        self._img_obs = None

    def observation(self, obs):
        img_obs = self.env.render(mode="rgb_array")['rgb_obs'] #'rgb_obs': rgb_obs, 'depth_obs': depth_obs}
        img_obs = img_obs[0][:, :, ::-1] #to get static camera
        img_obs = self.img_preprocessing(img_obs)
        full_obs, obs_dict = self.env.get_state_obs()#remove door states
        obs = {"rgb_obs": img_obs,
                "position": obs_dict['robot_obs']
                }
        self.obs_count +=1
        return obs

    def img_preprocessing(self, frame): #obs is from 0-255, (300,300,3)
        new_frame = cv2.resize(
            frame, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA) #100,100
        new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
        #cv2.normalize(new_frame, new_frame, 0, 255, norm_type = cv2.NORM_MINMAX)
        # cv2.imshow("win",new_frame)
        # cv2.waitKey(1)
        new_frame = np.expand_dims(new_frame, 0)#1,100,100
        if(self.obs_count == 0): #pos 0 is the most recent one
            zeros = np.zeros((self.history_length-1,self.img_size,self.img_size))
            self._img_obs = np.concatenate((new_frame, zeros), axis=0)
        elif(self.obs_count%(self.skip_frames+1)==0):
            self._img_obs = np.pad(self._img_obs,((1,0),(0,0),(0,0)), mode='constant')[:-1, :, :]
            self._img_obs[0] = new_frame
        return self._img_obs