# from torch.utils.data import DataLoader
import hydra
import os
import cv2
from hydra.utils import get_original_cwd, to_absolute_path
import torch
from omegaconf.listconfig import ListConfig
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import json
from vapo.env_wrappers.utils import load_aff_from_hydra

import vapo.utils.flowlib as flowlib
from vapo.utils.img_utils import overlay_mask, overlay_flow
from vapo.utils.file_manipulation import get_files
from vapo.affordance_model.datasets import get_transforms


def get_filenames(data_dir, get_eval_files=False, cam_type='gripper_cam'):
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
        if(get_eval_files):
            files, np_comprez = get_validation_files(data_dir, cam_type)
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


# Load validation files for custom datase
def get_validation_files(data_dir, cam_type):
    data_dir = os.path.join(get_original_cwd(), data_dir)
    data_dir = os.path.abspath(data_dir)
    json_file = os.path.join(data_dir,
                             "episodes_split.json")
    with open(json_file) as f:
        data = json.load(f)
    d = []
    for ep, imgs in data['validation'].items():
        im_lst = [data_dir + "/%s/data/%s.npz" % (ep, img_path)
                  for img_path in imgs if cam_type in img_path]
        d.extend(im_lst)
    return d, True


@hydra.main(config_path="../config", config_name="viz_affordances")
def viz(cfg):
    # Create output directory if save_images
    if(not os.path.exists(cfg.output_dir) and cfg.save_images):
        os.makedirs(cfg.output_dir)
    model, run_cfg = load_aff_from_hydra(cfg)
    model_cfg = run_cfg.model_cfg

    # Transforms
    cam_type = cfg.dataset.cam
    n_classes = model_cfg.n_classes
    cm = plt.get_cmap('tab10')
    transforms_cfg = run_cfg.dataset.transforms_cfg
    img_size = run_cfg.img_size[cam_type]
    img_transform = get_transforms(transforms_cfg.validation, img_size)
    # mask_transforms = get_transforms(transforms_cfg[_masks_t], img_size)

    # Iterate images
    files, np_comprez = get_filenames(cfg.data_dir,
                                      get_eval_files=cfg.get_eval_files,
                                      cam_type=cam_type)
    out_shape = (cfg.out_size, cfg.out_size)
    im_size = (run_cfg.img_size[cam_type], run_cfg.img_size[cam_type])
    for filename in tqdm.tqdm(files):
        if(np_comprez):
            data = np.load(filename)
            orig_img = data["frame"]
            gt_mask = data["mask"]
            gt_directions = data["directions"]
        else:
            orig_img = cv2.imread(filename, cv2.COLOR_BGR2RGB)
            out_shape = np.shape(orig_img)[:2]

        orig_img_resize = cv2.resize(orig_img, im_size)
        # Apply validation transforms
        x = torch.from_numpy(orig_img_resize).permute(2, 0, 1).unsqueeze(0)
        x = img_transform(x).cuda()

        # Predict affordance, centers and directions
        aff_logits, _, aff_mask, directions = model(x)
        fg_mask, _, object_centers, object_masks = \
            model.predict(aff_mask, directions)

        # if(np_comprez):
        #     gt_transformed = mask_transforms(
        #               torch.Tensor(np.expand_dims(gt_mask, 0)).cuda())
        #     # print(compute_mIoU(aff_logits, gt_transformed))

        # To numpy arrays
        pred_shape = np.array(fg_mask[0].shape)
        mask = fg_mask.detach().cpu().numpy()
        object_masks = object_masks.permute((1, 2, 0)).detach().cpu().numpy()
        directions = directions[0].detach().cpu().numpy()

        # Plot different objects according to voting layer
        obj_segmentation = orig_img_resize
        centers = []
        obj_class = np.unique(object_masks)[1:]
        obj_class = obj_class[obj_class != 0]  # remove background class
        colors = cm(np.linspace(0, 1, len(obj_class)))[:, :3]
        colors = (colors * 255).astype('uint8')
        for i, o in enumerate(object_centers):
            o = o.detach().cpu().numpy()
            centers.append(o)
            # obj_mask = np.zeros_like(object_masks)  # (img_size, img_size, 1)
            # obj_mask[object_masks == obj_class[i]] = 255
            # new_size = obj_segmentation.shape[:2]
            # resize_mask = cv2.resize(obj_mask, (*new_size, 1))
            # obj_segmentation = overlay_mask(resize_mask[:, :, 0],
            #                                 obj_segmentation,
            #                                 tuple(colors[i]))
        # Affordance segmentation
        affordances = orig_img
        if(n_classes > 2):
            # Not showing background
            colors = cm(np.linspace(0, 1, n_classes-1))[:, :3]
            colors = (colors[:, ::-1] * 255).astype('uint8')
            for i in range(1, n_classes):
                obj_mask = np.zeros_like(mask)  # (1, img_size, img_size)
                obj_mask[mask == i] = 255
                new_size = affordances.shape[:2]
                resize_mask = cv2.resize(obj_mask, new_size)
                affordances = overlay_mask(resize_mask[0],
                                           affordances,
                                           tuple(colors[i-1]))
            mask[mask > 0] = 1
        else:
            bin_mask = (mask[0] * 255).astype('uint8')
            bin_mask = cv2.resize(bin_mask, affordances.shape[:2])
            affordances = overlay_mask(bin_mask, affordances, (255, 0, 0))

        # To flow img
        directions = np.transpose(directions, (1, 2, 0))
        flow_img = flowlib.flow_to_image(directions)  # RGB
        flow_img = flow_img[:, :, ::-1]  # BGR
        mask = (np.transpose(mask, (1, 2, 0))*255).astype('uint8')
        # mask[:, 100:] = 0

        # Resize to out_shape
        obj_segmentation = cv2.resize(obj_segmentation, out_shape)
        orig_img = cv2.resize(orig_img, out_shape)
        flow_img = cv2.resize(flow_img, out_shape)
        mask = cv2.resize(mask, out_shape)
        affordances = cv2.resize(affordances, out_shape)

        # GT processing if available
        if(np_comprez):
            gt_flow = flowlib.flow_to_image(gt_directions)[:, :, ::-1]
            if(gt_mask.max() >= 1):  # multiclass
                gt_mask[gt_mask >= 1] = 255
            gt_mask = cv2.resize(gt_mask.astype('uint8'), out_shape)
            gt_directions = cv2.resize(gt_flow, out_shape)
            gt_res = overlay_flow(gt_directions, orig_img, gt_mask)

        # Resize centers
        new_shape = np.array(out_shape)
        for i in range(len(centers)):
            centers[i] = (centers[i] * new_shape / pred_shape).astype("int32")

        # Overlay directions and centers
        res = overlay_flow(flow_img, orig_img, mask)

        # Draw detected centers
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
            split = tail.split('.')
            name = "".join(split[:-1])
            # ext = split[-1]
            output_file = os.path.join(cfg.output_dir, name + ".png")
            cv2.imwrite(output_file, res)
        if(cfg.imshow):
            cv2.imshow("Affordance masks", affordances)
            cv2.imshow("object masks", obj_segmentation)
            cv2.imshow("flow", flow_img)
            if(np_comprez):
                cv2.imshow("gt", gt_res)
                cv2.imshow("gt_flow", gt_flow)
            cv2.imshow("output", res)
            cv2.waitKey(0)


if __name__ == "__main__":
    viz()
