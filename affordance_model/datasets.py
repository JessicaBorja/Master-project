import numpy as np
import torch
import hydra
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import os
import json


class VREnvData(Dataset):
    # split = "train" or "validation"
    def __init__(self, root_dir, transforms, n_episodes=-1,
                 split="train", cam="static", log=None):
        self.log = log
        self.masks_dir = "%s/masks/" % root_dir
        self.frames_dir = "%s/frames/" % root_dir
        _ids = self.read_json(os.path.join(root_dir, "ep_data.json"))
        self.data = self._create_split(_ids, split, cam, n_episodes)
        self.transforms = self.get_transforms(transforms[split])
        self.mask_transforms = self.get_transforms(transforms['masks'])

    def _create_split(self, data, split, cam, n_episodes):
        n_val_ep = 1
        if(n_episodes > len(data)):
            n_episodes = len(data)

        if(split == "train"):
            n_train_ep = n_episodes - n_val_ep
            valid_train_ep = len(data) - n_val_ep
            d_episodes = np.random.choice(
                            valid_train_ep, n_train_ep, replace=False)
        else:
            d_episodes = list(range(len(data) - n_val_ep, len(data)))

        split_data = []
        for ep, (_, file_lst) in enumerate(data.items()):
            for file in file_lst:
                if(ep in d_episodes and cam in file):
                    split_data.append(file)

        self.log.info("%s episodes: %s \t[%d] images"
                      % (split, str(d_episodes), len(split_data)))

        return split_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename = self.data[idx]
        frame = cv2.imread(
            self.frames_dir + filename + ".jpg",
            cv2.COLOR_BGR2RGB)
        frame = torch.from_numpy(frame).permute(2, 0, 1)  # C, W, H
        # frame = Image.open(self.frames_dir + filename  + ".jpg")
        frame = self.transforms(frame)

        # Segmentation mask
        mask = np.load(self.masks_dir + filename + ".npy")  # (H, W)
        # Resize from torchvision requires mask to have channel dim
        mask = np.expand_dims(mask, 0)
        mask = self.mask_transforms(torch.from_numpy(mask))
        # CE Loss requires mask in form (B, H, W), so remove channel dim
        mask = mask.squeeze()
        return frame, mask

    def get_transforms(self, transforms_cfg):
        transforms_lst = []
        for cfg in transforms_cfg:
            transforms_lst.append(hydra.utils.instantiate(cfg))
        return transforms.Compose(transforms_lst)

    def read_json(self, json_file):
        with open(json_file) as f:
            data = json.load(f)
        return data
