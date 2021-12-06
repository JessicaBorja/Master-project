# from torch.utils.data import DataLoader
import hydra
import os
import cv2
from hydra.utils import get_original_cwd, to_absolute_path
import torch
from omegaconf.listconfig import ListConfig
import tqdm
import numpy as np
import json
from vapo.env_wrappers.utils import load_aff_from_hydra

from vapo.utils.img_utils import transform_and_predict, get_aff_imgs
from vapo.utils.file_manipulation import get_files
from vapo.affordance_model.datasets import get_transforms
from vapo.utils.utils import torch_to_numpy


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
    transforms_cfg = run_cfg.dataset.transforms_cfg
    img_size = run_cfg.img_size[cam_type]
    img_transform = get_transforms(transforms_cfg.validation, img_size)

    # Iterate images
    files, np_comprez = get_filenames(cfg.data_dir,
                                      get_eval_files=cfg.get_eval_files,
                                      cam_type=cam_type)
    out_shape = (cfg.out_size, cfg.out_size)
    for filename in tqdm.tqdm(files):
        if(np_comprez):
            data = np.load(filename)
            orig_img = data["frame"]
            gt_mask = data["mask"]
            gt_directions = data["directions"]
        else:
            orig_img = cv2.imread(filename, cv2.COLOR_BGR2RGB)
            out_shape = np.shape(orig_img)[:2]

        res = transform_and_predict(model,
                                    img_transform,
                                    orig_img)
        centers, mask, directions, probs, _ = res
        affordances, _, flow_over_img, flow_img = \
            get_aff_imgs(orig_img, mask,
                         directions, centers,
                         out_shape,
                         cam=cam_type,
                         n_classes=probs.shape[-1])
        # Ground truth
        gt_directions = torch.tensor(gt_directions).permute(2, 0, 1)
        gt_directions = gt_directions.unsqueeze(0).contiguous().float().cuda()
        gt_centers, gt_directions, _ = model.get_centers(torch.tensor(gt_mask).unsqueeze(0).cuda(),
                                                         gt_directions)
        gt_directions = torch_to_numpy(gt_directions[0].permute(1, 2, 0))
        gt_aff, _, gt_res, gt_flow = get_aff_imgs(orig_img, gt_mask // 255,
                                                  gt_directions,
                                                  data['centers'],
                                                  out_shape,
                                                  cam=cam_type,
                                                  n_classes=model_cfg.n_classes)
        # Save and show
        if(cfg.save_images):
            _, tail = os.path.split(filename)
            split = tail.split('.')
            name = "".join(split[:-1])
            # ext = split[-1]
            output_file = os.path.join(cfg.output_dir, name + ".png")
            cv2.imwrite(output_file, flow_over_img)
        if(cfg.imshow):
            cv2.imshow("Affordance masks", affordances)
            cv2.imshow("flow", flow_img)
            if(np_comprez):
                cv2.imshow("gt", gt_res)
            cv2.imshow("output", flow_over_img)
            cv2.waitKey(0)


if __name__ == "__main__":
    viz()
