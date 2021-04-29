import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import os
import json
import hydra
from torch.utils.data import DataLoader
from torchvision import transforms
import sys
parent_dir = os.path.dirname(os.getcwd())
sys.path.insert(0, os.getcwd())
sys.path.insert(0, parent_dir)
sys.path.insert(0, parent_dir+"/VREnv/")


def get_transforms(transforms_cfg):
    transforms_lst = []
    for cfg in transforms_cfg:
        transforms_lst.append(hydra.utils.instantiate(cfg))
    return transforms.Compose(transforms_lst)


def get_loaders(logger, dataset_cfg, dataloader_cfg):
    train = VREnvData(split="train", log=logger, **dataset_cfg)
    val = VREnvData(split="validation", log=logger, **dataset_cfg)
    logger.info('train_data {}'.format(train.__len__()))
    logger.info('val_data {}'.format(val.__len__()))

    train_loader = DataLoader(train, shuffle=True, **dataloader_cfg)
    val_loader = DataLoader(val, **dataloader_cfg)
    logger.info('train minibatches {}'.format(len(train_loader)))
    logger.info('val minibatches {}'.format(len(val_loader)))
    return train_loader, val_loader


class VREnvData(Dataset):
    # split = "train" or "validation"
    def __init__(self, root_dir, transforms, n_train_ep=-1,
                 split="train", cam="static", log=None):
        self.log = log
        self.root_dir = root_dir
        _ids = self.read_json(os.path.join(root_dir, "episodes_split.json"))
        self.data = self._get_split_data(_ids, split, cam, n_train_ep)
        self.transforms = get_transforms(transforms[split])
        self.mask_transforms = get_transforms(transforms['masks'])

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
        head, filename = os.path.split(self.data[idx].replace("\\", "/"))
        episode, cam_folder = os.path.normpath(head).split(os.path.sep)
        data = np.load(self.root_dir +
                       "/%s/data/%s/%s.npz" % (episode, cam_folder, filename))

        # Images are stored in BGR
        frame = data["frame"]
        frame = torch.from_numpy(frame).permute(2, 0, 1)  # C, W, H
        frame = self.transforms(frame)

        # Segmentation mask (H, W)
        mask = data["mask"]
        # Resize from torchvision requires mask to have channel dim
        mask = np.expand_dims(mask, 0)
        mask = self.mask_transforms(torch.from_numpy(mask))
        # CE Loss requires mask in form (B, H, W), so remove channel dim
        mask = mask.squeeze()
        return frame, mask

    def read_json(self, json_file):
        with open(json_file) as f:
            data = json.load(f)
        return data


@hydra.main(config_path="../config", config_name="cfg_affordance")
def main(cfg):
    val = VREnvData(split="validation", log=None,
                    **cfg.dataset)
    val_loader = DataLoader(val, num_workers=4, batch_size=1, pin_memory=True)
    print('val minibatches {}'.format(len(val_loader)))

    for i, item in enumerate(val_loader):
        frame, _ = item
        frame = frame[0].detach().cpu().numpy()
        cv2.imshow("img", frame)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
