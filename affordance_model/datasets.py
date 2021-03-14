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
    def __init__(self, root_dir, transforms, n_train_ep=-1,
                 split="train", cam="static", log=None):
        self.log = log
        self.root_dir = root_dir
        _ids = self.read_json(os.path.join(root_dir, "episodes_split.json"))
        self.data = self._get_split_data(_ids, split, cam, n_train_ep)
        self.transforms = self.get_transforms(transforms[split])
        self.mask_transforms = self.get_transforms(transforms['masks'])

    def _get_split_data(self, data, split, cam, n_train_ep):
        split_data = []
        split_episodes = list(data[split].keys())

        # Select amount of data to train on
        if(n_train_ep > 0 and split == "train"):
            assert len(split_episodes) >= n_train_ep, \
                "n_train_ep must <= %d" % len(split_episodes)
            split_episodes = np.random.choice(split_episodes,
                                              n_train_ep,
                                              replace=False)

        print("%s episodes: %s" % (split, str(split_episodes)))
        for ep in split_episodes:
            for file in data[split][ep]:
                if(cam in file):
                    split_data.append("%s/%s" % (ep, file))
        print("%s images: %d" % (split, len(split_data)))
        return split_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        head, filename = os.path.split(self.data[idx])
        episode, cam_folder = os.path.normpath(head).split(os.path.sep)
        frame = cv2.imread(self.root_dir +
                           "/%s/frames/%s/%s.jpg" % (episode, cam_folder, filename),
                           cv2.COLOR_BGR2RGB)
        frame = torch.from_numpy(frame).permute(2, 0, 1)  # C, W, H
        # frame = Image.open(self.frames_dir + filename  + ".jpg")
        frame = self.transforms(frame)

        # Segmentation mask (H, W)
        mask = np.load(self.root_dir +
                       "/%s/masks/%s/%s.npy" % (episode, cam_folder, filename))
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
