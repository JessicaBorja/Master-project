import numpy as np
import torch
import hydra
from torch.utils.data import Dataset
from torchvision import transforms, utils
import cv2
import os
import json
from PIL import Image

class VREnvData(Dataset):
    def __init__(self, root_dir, transforms, split="train"): #split = "train" or "validation"
        self.masks_dir = "%s/masks/"%root_dir
        self.frames_dir = "%s/frames/"%root_dir
        _ids = self.read_json(os.path.join(root_dir, "data.json"))
        #self.data = _ids[split]
        self.data = [x for x in _ids[split] if "gripper" not in x]
        self.transforms = self.get_transforms(transforms[split])
        self.mask_transforms = self.get_transforms(transforms['masks'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename = self.data[idx]
        frame = cv2.imread( self.frames_dir + filename  + ".jpg", cv2.COLOR_BGR2RGB)
        frame = torch.from_numpy(frame).permute(2,0,1) #C, W, H
        #frame = Image.open(self.frames_dir + filename  + ".jpg")
        frame = self.transforms(frame)
        
        # Segmentation mask
        mask = np.load( self.masks_dir + filename + ".npy" ) # (H, W)
        # Resize from torchvision requires mask to have channel dim
        mask = np.expand_dims(mask, 0)
        mask = self.mask_transforms( torch.from_numpy(mask) )
        # CE Loss requires mask in form (B, H, W), so remove channel dim
        mask = mask.squeeze()
        return frame, mask

    def get_transforms(self, transforms_cfg):
        transforms_lst = []
        for cfg in transforms_cfg:
            transforms_lst.append( hydra.utils.instantiate(cfg) )
        return transforms.Compose(transforms_lst)

    def read_json(self, json_file):
        with open(json_file) as f:
            data = json.load(f)
        return data