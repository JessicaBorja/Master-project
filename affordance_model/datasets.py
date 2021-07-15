import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import os
import json
import hydra
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import rotate
import sys
parent_dir = os.path.dirname(os.getcwd())
sys.path.insert(0, os.getcwd())
sys.path.insert(0, parent_dir)
sys.path.insert(0, parent_dir+"/VREnv/")
import utils.flowlib as flowlib
from utils.img_utils import overlay_flow, overlay_mask, tresh_np
from sklearn.cluster import DBSCAN


def get_transforms(transforms_cfg, img_size=None, add_rotation=False):
    transforms_lst = []
    transforms_config = transforms_cfg.copy()
    for cfg in transforms_config:
        if(cfg._target_ == "torchvision.transforms.Resize"
           and img_size is not None):
            cfg.size = img_size
        transforms_lst.append(hydra.utils.instantiate(cfg))

    return transforms.Compose(transforms_lst)


def get_loaders(logger, dataset_cfg, dataloader_cfg, img_size, n_classes):
    train = VREnvData(img_size,
                      split="train",
                      log=logger,
                      n_classes=n_classes,
                      **dataset_cfg)
    val = VREnvData(img_size,
                    split="validation",
                    log=logger,
                    n_classes=n_classes,
                    **dataset_cfg)
    logger.info('train_data {}'.format(train.__len__()))
    logger.info('val_data {}'.format(val.__len__()))

    train_loader = DataLoader(train, shuffle=True, **dataloader_cfg)
    val_loader = DataLoader(val, **dataloader_cfg)
    logger.info('train minibatches {}'.format(len(train_loader)))
    logger.info('val minibatches {}'.format(len(val_loader)))
    return train_loader, val_loader


class VREnvData(Dataset):
    # split = "train" or "validation"
    def __init__(self, img_size, root_dir, transforms_cfg, n_train_ep=-1,
                 split="train", cam="static", log=None, n_classes=2):
        self.cam = cam
        self.split = split
        self.log = log
        self.root_dir = root_dir
        _ids = self.read_json(os.path.join(root_dir, "episodes_split.json"))
        self.data = self._get_split_data(_ids, split, cam, n_train_ep)
        self.add_rotation = split == "train" and cam == "gripper"
        self.transforms = get_transforms(transforms_cfg[split], img_size,
                                         add_rotation=self.add_rotation)
        _masks_t = "masks" if n_classes <= 2 else "masks_multitask"
        self.mask_transforms = get_transforms(transforms_cfg[_masks_t], img_size)
        self.pixel_indices = np.indices((img_size, img_size),
                                        dtype=np.float32).transpose(1, 2, 0)
        self.img_size = img_size

    def _overfit_split_data(self, data, split, cam, n_train_ep):
        split_data = []
        split_episodes = ["episode_0"]

        print("%s episodes: %s" % (split, str(split_episodes)))
        for ep in split_episodes:
            for file in data[split][ep]:
                if(cam in file or cam == "full"):
                    split_data.append("%s/%s" % (ep, file))
        print("%s images: %d" % (split, len(split_data)))
        return split_data

    def _get_split_data(self, data, split, cam, n_train_ep):
        split_data = []
        split_episodes = list(data[split].keys())

        # Select amount of data to train on
        if(n_train_ep > 0
           and split == "train"):
            assert len(split_episodes) >= n_train_ep, \
                "n_train_ep must <= %d" % len(split_episodes)
            split_episodes = np.random.choice(split_episodes,
                                              n_train_ep,
                                              replace=False)

        print("%s episodes: %s" % (split, str(split_episodes)))
        for ep in split_episodes:
            for file in data[split][ep]:
                if(cam in file or cam == "full"):
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
            mid_point = mid_point.astype('uint8')  # px coords
            center = np.array([mid_point[1], mid_point[0]])  # matrix coords
            centers.append(center)

            # Object mask
            object_mask = np.zeros_like(mask)
            object_mask[cluster[:, 0], cluster[:, 1]] = 1
            object_center_directions = \
                (mid_point - self.pixel_indices).astype(np.float32)
            object_center_directions = object_center_directions\
                / np.maximum(np.linalg.norm(object_center_directions,
                                            axis=2, keepdims=True), 1e-10)

            # Add it to the labels
            directions[object_mask == 1] = \
                object_center_directions[object_mask == 1]

        directions = np.transpose(directions, (2, 0, 1))
        return centers, directions  # 2, H, W

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # directions: optical flow image in middlebury color

        head, filename = os.path.split(self.data[idx].replace("\\", "/"))
        episode, cam_folder = os.path.split(head)
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
        mask = self.mask_transforms(torch.from_numpy(mask)).long()

        # centers, center_dirs = self.get_directions(mask)
        center_dirs = torch.tensor(data["directions"]).permute(2, 0, 1)

        # Rotation transform
        if(self.cam == "gripper" and self.split == "train"):
            rand_angle = np.random.randint(-180, 180)
            frame = rotate(frame, rand_angle)
            mask = rotate(mask, rand_angle)
            center_dirs = rotate(center_dirs, rand_angle)

        # CE Loss requires mask in form (B, H, W), so remove channel dim
        mask = mask.squeeze()  # H, W

        labels = {"affordance": mask,
                  # "centers": centers,
                  "center_dirs": center_dirs}
        return frame, labels

    def read_json(self, json_file):
        with open(json_file) as f:
            data = json.load(f)
        return data


def test_dir_labels(hv, frame, aff_mask, center_dir):
    # center_dir /= torch.norm(center_dir,
    #                          dim=1,
    #                          keepdim=True
    #                          ).clamp(min=1e-10)
    flow_img = center_dir[0].permute((1, 2, 0)).cpu().detach().numpy()
    flow_img = flowlib.flow_to_image(flow_img)  # RGB
    flow_img = flow_img[:, :, ::-1]  # BGR
    cv2.imshow("directions_l", flow_img)
    cv2.waitKey(1)
    bool_mask = (aff_mask == 1).int().cuda()
    center_dir = center_dir.cuda()  # 1, 2, H, W
    initial_masks, num_objects, object_centers_padded = \
        hv(bool_mask, center_dir.contiguous())

    initial_masks = initial_masks.cpu()
    object_centers_padded = object_centers_padded[0].cpu().permute((1, 0))
    for c in object_centers_padded:
        c = c.detach().cpu().numpy()
        u, v = int(c[1]), int(c[0])  # center stored in matrix convention
        frame = cv2.drawMarker(frame, (u, v),
                               (0, 0, 0),
                               markerType=cv2.MARKER_CROSS,
                               markerSize=5,
                               line_type=cv2.LINE_AA)
    cv2.imshow("hv_center", frame)
    cv2.waitKey(1)
    return frame


@hydra.main(config_path="../config", config_name="cfg_affordance")
def main(cfg):
    img_size = cfg.img_size[cfg.dataset.cam]
    val = VREnvData(img_size, split="train", log=None,
                    **cfg.dataset)
    val_loader = DataLoader(val, num_workers=1, batch_size=1, pin_memory=True)
    print('val minibatches {}'.format(len(val_loader)))
    from affordance_model.hough_voting import hough_voting as hv
    hv = hv.HoughVoting(**cfg.model_cfg.hough_voting)

    for b_idx, b in enumerate(val_loader):
        frame, labels = b
        directions = labels["center_dirs"][0].detach().cpu().numpy()
        mask = labels["affordance"].detach().cpu().numpy()

        frame = frame[0].detach().cpu().numpy()
        frame = ((frame + 1)*255/2).astype('uint8')
        frame = np.transpose(frame, (1, 2, 0))
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        directions = np.transpose(directions, (1, 2, 0))
        flow_img = flowlib.flow_to_image(directions)  # RGB
        flow_img = flow_img[:, :, ::-1]  # BGR
        mask = np.transpose(mask, (1, 2, 0))*255
        out_img = overlay_flow(flow_img, frame, mask)

        out_img = test_dir_labels(hv,
                                  out_img,
                                  labels["affordance"],
                                  labels["center_dirs"])


        # centers = labels["centers"][0]
        # for c in centers:
        #     c = c.squeeze().detach().cpu().numpy()
        #     u, v = c[1], c[0]  # center stored in matrix convention
        #     out_img = cv2.drawMarker(out_img, (u, v),
        #                              (0, 0, 0),
        #                              markerType=cv2.MARKER_CROSS,
        #                              markerSize=5,
        #                              line_type=cv2.LINE_AA)
        out_img = cv2.resize(out_img, (200, 200),
                             interpolation=cv2.INTER_CUBIC)
        cv2.imshow("img", out_img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
