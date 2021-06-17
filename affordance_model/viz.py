# from torch.utils.data import DataLoader
import hydra
import os
import cv2
from hydra.utils import get_original_cwd, to_absolute_path
import torch
import sys
from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig
import tqdm
import numpy as np
parent_dir = os.path.dirname(os.getcwd())
sys.path.insert(0, os.getcwd())
sys.path.insert(0, parent_dir)
import utils.flowlib as flowlib
from utils.img_utils import overlay_mask, visualize, overlay_flow
from utils.file_manipulation import get_files
from affordance_model.segmentator_centers import Segmentator
from affordance_model.datasets import get_transforms
from affordance_model.utils.losses import compute_mIoU
from matplotlib import cm
import json


def get_filenames(data_dir):
    files = []
    np_comprez = False
    if(isinstance(data_dir, ListConfig)):
        for dir_i in data_dir:
            dir_i = to_absolute_path(dir_i)
            dir_i = os.path.abspath(dir_i)
            if(not os.path.exists(dir_i)):
                print("Path does not exist: %s" % dir_i)
                continue
            files += get_files(dir_i, "npz")
            if(len(files) > 0):
                np_comprez = True
            files += get_files(dir_i, "jpg")
            files += get_files(dir_i, "png")
    else:
        data_dir = to_absolute_path(data_dir)
        data_dir = os.path.abspath(data_dir)
        if(not os.path.exists(data_dir)):
            print("Path does not exist: %s" % data_dir)
            return
        files += get_files(data_dir, "npz")
        if(len(files) > 0):
            np_comprez = True
        files += get_files(data_dir, "jpg")
        files += get_files(data_dir, "png")
    return files, np_comprez


def get_validation_files(data_dir):
    json_file = os.path.join(data_dir[0], "episodes_split.json")
    with open(json_file) as f:
        data = json.load(f)
    d = []
    for e in data['validation']['episode_1']:
        cam_folder, filename = os.path.split(e.replace("\\", "/"))
        d.append(data_dir[0] + "/%s/data/%s/%s.npz" % ("episode_1", cam_folder, filename))
    return d, True


@hydra.main(config_path="../config", config_name="viz_affordances")
def viz(cfg):
    # Create output directory if save_images
    if(not os.path.exists(cfg.output_dir) and cfg.save_images):
        os.makedirs(cfg.output_dir)
    # Initialize model
    run_cfg = OmegaConf.load(cfg.folder_name + "/.hydra/config.yaml")
    model_cfg = run_cfg.model_cfg
    model_cfg.hough_voting = cfg.model_cfg.hough_voting

    checkpoint_path = os.path.join(cfg.folder_name, "trained_models")
    checkpoint_path = os.path.join(checkpoint_path, cfg.model_name)
    model = Segmentator.load_from_checkpoint(checkpoint_path, cfg=model_cfg).cuda()
    model.eval()
    print("model loaded")

    # Image needs to be multiple of 32 because of skip connections
    # and decoder layers
    img_transform = get_transforms(cfg.transforms.validation)
    mask_transforms = get_transforms(cfg.transforms.masks)
    # Iterate images
    files, np_comprez = get_filenames(cfg.data_dir)
    n = len(files) // 2
    files = files[n:]
    out_shape = (cfg.out_size, cfg.out_size)

    for filename in tqdm.tqdm(files):
        if(np_comprez):
            data = np.load(filename)
            orig_img = data["frame"]
            gt_mask = data["mask"]
            gt_directions = data["directions"]
        else:
            orig_img = cv2.imread(filename, cv2.COLOR_BGR2RGB)

        # Apply validation transforms
        x = torch.from_numpy(orig_img).permute(2, 0, 1).unsqueeze(0)
        x = img_transform(x).cuda()

        # Predict affordance, centers and directions
        aff_logits, _, aff_mask, directions = model(x)
        fg_mask, _, object_centers, object_masks = \
            model.predict(aff_mask, directions)

        gt_transformed = mask_transforms(torch.Tensor(np.expand_dims(gt_mask, 0)).cuda())
        # print(compute_mIoU(aff_logits, gt_transformed))

        # To numpy arrays
        pred_shape = np.array(fg_mask[0].shape)
        mask = fg_mask.detach().cpu().numpy()
        object_masks = object_masks.permute((1, 2, 0)).detach().cpu().numpy()
        directions = directions[0].detach().cpu().numpy()

        # Plot different objects
        centers = []
        obj_segmentation = orig_img
        obj_class = np.unique(object_masks)[1:]
        obj_class = obj_class[obj_class != 0]  # remove background class
        colors = cm.jet(np.linspace(0, 1, len(obj_class)))
        for i, o in enumerate(object_centers):
            o = o.detach().cpu().numpy()
            centers.append(o)
            obj_mask = np.zeros_like(object_masks)
            obj_mask[object_masks == obj_class[i]] = 255
            obj_segmentation = overlay_mask(obj_mask[:, :, 0],
                                            obj_segmentation,
                                            tuple([int(c*255) for c in colors[i][:3]]))

        # To flow img
        directions = np.transpose(directions, (1, 2, 0))
        flow_img = flowlib.flow_to_image(directions)  # RGB
        flow_img = flow_img[:, :, ::-1]  # BGR
        gt_flow = flowlib.flow_to_image(gt_directions)[:, :, ::-1]

        mask = (np.transpose(mask, (1, 2, 0))*255).astype('uint8')

        # Resize to out_shape
        obj_segmentation = cv2.resize(obj_segmentation, out_shape)
        orig_img = cv2.resize(orig_img, out_shape)
        flow_img = cv2.resize(flow_img, out_shape)
        mask = cv2.resize(mask, out_shape)
        gt_mask = cv2.resize(gt_mask.astype('uint8'), out_shape)
        gt_directions = cv2.resize(gt_flow, out_shape)

        # Resize centers
        new_shape = np.array(out_shape)
        for i in range(len(centers)):
            centers[i] = (centers[i] * new_shape / pred_shape).astype("int32")

        # Overlay directions and centers
        res = overlay_flow(flow_img, orig_img, mask)
        gt_res = overlay_flow(gt_directions, orig_img, gt_mask)
        for c in centers:
            u, v = c[1], c[0]  # center stored in matrix convention
            res = cv2.drawMarker(res, (u, v),
                                 (0, 0, 0),
                                 markerType=cv2.MARKER_CROSS,
                                 markerSize=10,
                                 line_type=cv2.LINE_AA)

        # Save and show
        if(cfg.save_images):
            _, tail = os.path.split(filename)
            name, ext = tail.split('.')
            output_file = os.path.join(cfg.output_dir, name + ".jpg")
            cv2.imwrite(output_file, res)
        if(cfg.imshow):
            cv2.imshow("object masks", obj_segmentation)
            cv2.imshow("flow", flow_img)
            cv2.imshow("gt", gt_res)
            cv2.imshow("output", res)
            cv2.waitKey(1)


if __name__ == "__main__":
    viz()
