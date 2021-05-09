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
import utils.flowlib as flowlib
from utils.img_utils import overlay_flow, tresh_np
from sklearn.cluster import DBSCAN


def get_transforms(transforms_cfg):
    transforms_lst = []
    for cfg in transforms_cfg:
        transforms_lst.append(hydra.utils.instantiate(cfg))
    return transforms.Compose(transforms_lst)


def get_loaders(logger, dataset_cfg, dataloader_cfg, img_size):
    train = VREnvData(img_size, split="train", log=logger, **dataset_cfg)
    val = VREnvData(img_size, split="validation", log=logger, **dataset_cfg)
    logger.info('train_data {}'.format(train.__len__()))
    logger.info('val_data {}'.format(val.__len__()))

    train_loader = DataLoader(train, shuffle=True, **dataloader_cfg)
    val_loader = DataLoader(val, **dataloader_cfg)
    logger.info('train minibatches {}'.format(len(train_loader)))
    logger.info('val minibatches {}'.format(len(val_loader)))
    return train_loader, val_loader


class VREnvData(Dataset):
    # split = "train" or "validation"
    def __init__(self, img_size, root_dir, transforms, n_train_ep=-1,
                 split="train", cam="static", log=None):
        self.log = log
        self.root_dir = root_dir
        _ids = self.read_json(os.path.join(root_dir, "episodes_split.json"))
        self.data = self._get_split_data(_ids, split, cam, n_train_ep)
        self.transforms = get_transforms(transforms[split])
        self.mask_transforms = get_transforms(transforms['masks'])
        self.pixel_indices = np.indices((img_size, img_size),
                                        dtype=np.float32).transpose(1, 2, 0)
        self.img_size = img_size

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

    def get_directions(self, mask):
        # Mask.shape = img_size, img_size between 0-1
        dbscan = DBSCAN(eps=3, min_samples=3)
        mask = mask.detach().cpu().numpy()
        positives = np.argwhere(mask > 0.5)
        directions = np.stack([np.ones((self.img_size, self.img_size)),
                              np.zeros((self.img_size, self.img_size))],
                              axis=-1).astype(np.float32)
        centers = []
        if(positives.shape[0] > 0):
            labels = dbscan.fit_predict(positives)
        else:
            return centers, directions

        for idx, c in enumerate(np.unique(labels)):
            cluster = positives[np.argwhere(labels == c).squeeze()]  # N, 3
            if(len(cluster.shape) == 1):
                cluster = np.expand_dims(cluster, 0)
            mid_point = np.mean(cluster, 0)[:2]
            mid_point = mid_point.astype('uint8')
            center = np.array([mid_point[1], mid_point[0]])
            centers.append(center)
            # Object mask
            object_mask = np.zeros_like(mask)
            object_mask[cluster] = 1
            object_center_directions = \
                (center - self.pixel_indices).astype(np.float32)
            object_center_directions = object_center_directions\
                / np.maximum(np.linalg.norm(object_center_directions,
                                            axis=2, keepdims=True), 1e-10)

            # Add it to the labels
            directions[object_mask == 1] = \
                object_center_directions[object_mask == 1]
        return centers, directions

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # directions: optical flow image in middlebury color

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
        mask = mask.squeeze()  # H, W

        # centers, center_dirs = self.get_directions(mask)

        # centers = data["centers"]
        center_dirs = torch.tensor(data["directions"]).permute(2, 0, 1)
        # orig_frame = cv2.resize(data["frame"],
        #                         (self.img_size, self.img_size))
        labels = {"affordance": mask,
                  # "centers": centers,
                  "center_dirs": center_dirs}
        return frame, labels

    def read_json(self, json_file):
        with open(json_file) as f:
            data = json.load(f)
        return data


@hydra.main(config_path="../config", config_name="cfg_affordance")
def main(cfg):
    val = VREnvData(cfg.img_size, split="validation", log=None,
                    **cfg.dataset)
    val_loader = DataLoader(val, num_workers=4, batch_size=20, pin_memory=True)
    print('val minibatches {}'.format(len(val_loader)))

    for b_idx, b in enumerate(val_loader):
        frame, labels = b
        directions = labels["center_dirs"].detach().cpu().numpy()
        mask = labels["affordance"].detach().cpu().numpy()

        directions = np.transpose(directions, (1, 2, 0))
        flow_img = flowlib.flow_to_image(directions)  # RGB
        flow_img = flow_img[:, :, ::-1]  # BGR
        frame = frame.detach().cpu().numpy()
        frame = ((frame + 1)*255/2).astype('uint8')
        frame = np.transpose(frame, (1, 2, 0))
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        mask = np.transpose(mask, (1, 2, 0))*255

        out_img = overlay_flow(flow_img, frame, mask)
        centers = labels["centers"][0]
        for c in centers:
            c = c.squeeze().detach().cpu().numpy()
            u, v = c[1], c[0]  # center stored in matrix convention
            out_img = cv2.drawMarker(out_img, (u, v),
                                     (0, 0, 0),
                                     markerType=cv2.MARKER_CROSS,
                                     markerSize=5,
                                     line_type=cv2.LINE_AA)
        out_img = cv2.resize(out_img, (200, 200),
                             interpolation=cv2.INTER_CUBIC)
        cv2.imshow("img", out_img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
