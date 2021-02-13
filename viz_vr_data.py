from torch.utils.data import DataLoader
import hydra
from torchvision import transforms
from affordance_model.segmentator import Segmentator
import glob, os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from affordance_model.utils.utils import *
from affordance_model.utils.transforms import ScaleImageTensor
from torchvision import transforms
from torchvision.transforms import Resize
from omegaconf import OmegaConf
import pickle as pl

def viz_mask_img_pairs():
    path = "/mnt/16867D9A9C78B590/Users/Jessica/Documents/Proyecto_ssd/datasets/vrenv_playdata/"
    masks_dir = "%s/masks/"%path
    frames_dir = "%s/frames/"%path

    static_files = glob.glob(frames_dir + "/static_cam/*")
    static_files.sort()
    gripper_files = glob.glob(frames_dir + "/gripper_cam/*")
    gripper_files.sort()
    for abs_path in static_files:
        filename = os.path.basename(abs_path)[:-4] # Remove Extension
        cam_type = os.path.basename( os.path.split(abs_path)[0] )
        filename = os.path.join(cam_type,filename)
        frame = cv2.imread( frames_dir + filename  + ".jpg", cv2.COLOR_BGR2RGB)
        mask = np.load( masks_dir + filename + ".npy" )
        img = overlay_mask(mask, frame, (0,0,255))
        cv2.imshow("Gripper", img)
        cv2.waitKey(0)

def viz_rendered_data():
    #Iterate images
    path = "/mnt/16867D9A9C78B590/Users/Jessica/Documents/Proyecto_ssd/datasets/play_data/rendered_data/"
    files = glob.glob(path + "/*.npz")
    for idx, filename in enumerate(files):
        try:
            data = np.load(filename, allow_pickle=True)
            cv2.imshow("static", data['rgb_static'][:,:,::-1]) # W, H, C
            cv2.imshow("gripper", data['rgb_gripper'][:,:,::-1]) # W, H, C
            cv2.waitKey(0)
            print(data['actions']) # tcp pos(3), euler angles (3), gripper_action(0 close - 1 open)
        except:
            print("cannot load file as numpy compressed: %s"%filename)
        
if __name__ == "__main__":
    viz_mask_img_pairs()
    #viz_rendered_data()