from types import new_class
import hydra
import os
import sys
import cv2
import numpy as np
import tqdm
import json
import matplotlib.pyplot as plt
import pickle
parent_dir = os.path.dirname(os.getcwd())
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.getcwd())
sys.path.insert(0, parent_dir+"/VREnv/")
import utils.flowlib as flowlib
from utils.img_utils import overlay_mask, tresh_np, overlay_flow
from utils.label_segmentation import get_static_mask, get_gripper_mask
from utils.file_manipulation import get_files, save_data, check_file,\
                                    create_data_ep_split, merge_datasets


# Keep points in a distance larger than radius from new_point
# Do not keep fixed points more than 100 frames
def update_fixed_points(fixed_points, new_point,
                        current_frame_idx, radius=0.1):
    x = []
    for frame_idx, p, label in fixed_points:
        if np.linalg.norm(new_point - p) > radius:
            # and current_frame_idx - frame_idx < 1500:
            x.append((frame_idx, p, label))
    return x


def label_directions(center, object_mask, direction_labels):
    # Get directions
    # Shape: [H x W x 2]

    object_mask = tresh_np(object_mask, 100)
    object_center_directions = (center - pixel_indices).astype(np.float32)
    object_center_directions = object_center_directions\
        / np.maximum(np.linalg.norm(object_center_directions,
                                    axis=2, keepdims=True), 1e-10)

    # Add it to the labels
    direction_labels[object_mask == 1] = \
        object_center_directions[object_mask == 1]
    return direction_labels


def load_clustering(path):
    file = os.path.join(path, "k_means.pkl")
    k_means = pickle.load(open(file, 'rb'))
    centers = k_means.cluster_centers_
    n_classes = len(centers)
    cm = plt.get_cmap('tab10')
    colors = cm(np.linspace(0, 1, n_classes))[:, :3]
    # RGB -> BGR (open_cv)
    colors = (colors[:, ::-1] * 255).astype('uint8')
    return k_means, colors, n_classes


# Always classify as neighrest distance
def classify(point, trajectories):
    best_match = 0
    min_dist = 10000  # arbitrary high number
    for label, points in trajectories.items():
        # points (n,3), query_point = (3)
        curr_pt = np.expand_dims(point, 0)  # 1, 3
        distance = np.linalg.norm(np.array(points) - curr_pt, axis=-1)
        dist = min(distance)
        if(dist < min_dist):
            best_match = label
            min_dist = dist
    # Class 0 is background
    return int(best_match) + 1


def update_mask(fixed_points, mask, directions,
                frame_img_tuple, cam, out_img_size):
    # Update masks with fixed_points
    centers = []
    (frame_timestep, img) = frame_img_tuple
    for point_timestep, p, label in fixed_points:
        # Only add point if it was fixed before seing img
        if(frame_timestep >= point_timestep):
            new_mask, center_px = get_static_mask(cam, img, p)
            new_mask, center_px = resize_mask_and_center(new_mask, center_px,
                                                         out_img_size)
            centers.append(center_px)
            directions = label_directions(center_px, new_mask, directions)
            mask[label] = overlay_mask(new_mask, np.squeeze(mask[label]), (255, 255, 255))
    return mask, centers, directions


def create_gripper_cam_properties(cam_cfg):
    proj_m = {}
    cam_cfg = dict(cam_cfg)
    # Properties for projection matrix
    proj_m["fov"] = cam_cfg.pop("fov")
    proj_m["aspect"] = cam_cfg.pop("aspect")
    proj_m["nearVal"] = cam_cfg.pop("nearval")
    proj_m["farVal"] = cam_cfg.pop("farval")

    cam_cfg["proj_matrix"] = proj_m

    del_keys = ["_target_", "name", "gripper_cam_link"]
    for k in del_keys:
        cam_cfg.pop(k)
    return cam_cfg


def label_gripper(cam_properties, img_hist, point, viz,
                  save_dict, out_img_size):
    for idx, (fr_idx, im_id, robot_obs, img) in enumerate(img_hist):
        if(im_id not in save_dict):
            # Shape: [H x W x 2]
            H, W = out_img_size  # img.shape[:2]
            directions = np.stack(
                            [np.ones((H, W)),
                             np.zeros((H, W))], axis=-1).astype(np.float32)
            # Center and directions in matrix convention (row, column)
            mask, center_px = get_gripper_mask(img, robot_obs, point,
                                               cam_properties, radius=25)
            mask, center_px = resize_mask_and_center(mask, center_px,
                                                     out_img_size)
            directions = label_directions(center_px, mask, directions)

            # Visualize results
            img = cv2.resize(img, out_img_size)
            out_img = overlay_mask(mask, img, (0, 0, 255))
            flow_img = flowlib.flow_to_image(directions)[:, :, ::-1]
            flow_over_img = overlay_flow(flow_img, img, mask)

            if(viz):
                cv2.imshow("Gripper", out_img)
                cv2.imshow('Gripper flow_img', flow_over_img)
                cv2.imshow('Gripper real flow', flow_img)
                cv2.waitKey(1)

            save_dict[im_id] = {
                "frame": img,
                "mask": mask,
                "centers": np.stack([center_px]),
                "directions": directions,
                "viz_out": out_img,
                "viz_dir": flow_over_img}
    return save_dict


def resize_mask_and_center(mask, center, new_size):
    orig_H, orig_W = mask.shape[:2]
    mask = cv2.resize(mask, new_size)
    center = np.array(center) * new_size[0] // orig_H
    return mask, center


def label_static(static_cam, static_hist, back_max, occ_frames,
                 n_classes, colors,
                 fixed_points, pt, viz, save_dict, out_img_size):
    for idx, (fr_idx, im_id, img) in enumerate(static_hist):
        # For static mask assume oclusion
        # until back_frames_min before
        centers = []
        H, W = out_img_size  # img.shape[:2]  # img_shape = (H, W, C)
        directions = np.stack([np.ones((H, W)),
                               np.zeros((H, W))], axis=-1).astype(np.float32)
        # Class 0 is background
        full_mask, centers_px, fp_directions = update_mask(
            fixed_points,
            np.zeros((n_classes + 1, H, W)),
            directions,
            (fr_idx, img),
            static_cam,
            out_img_size)
        # first create fp masks and place current(newest)
        # mask and optical flow on top
        # Frame is before occlusion start
        if(fr_idx <= occ_frames[0] and
                idx > occ_frames[0] - back_max):
            # Get new grip
            mask, center_px = get_static_mask(static_cam, img, pt)
            mask, center_px = resize_mask_and_center(mask, center_px,
                                                     out_img_size)
            # These are center direction vectors
            directions = label_directions(center_px, mask, fp_directions)
            centers.append(center_px)
        else:
            # No segmentation in current image due to occlusion
            mask = np.zeros((H, W))

        # Concat to full mask
        # (frame_idx, point, class_label)
        # Last fixed point is the newest
        label = fixed_points[-1][-1]
        label_mask = np.stack((full_mask[label],
                               np.expand_dims(mask, 0))).any(axis=0).astype('uint8')
        full_mask[label] = label_mask * 255.0

        # Visualize
        img = cv2.resize(img, out_img_size)
        static_mask = full_mask.any(axis=0).astype('uint8')  # Binary
        flow_img = flowlib.flow_to_image(directions)[:, :, ::-1]
        flow_over_img = overlay_flow(flow_img, img, static_mask*255)

        # Color classes
        out_img = img
        # Start in 1, class 0 is background
        for label, (mask, c) in enumerate(zip(full_mask[1:], colors), 1):
            color = tuple(c)
            out_img = overlay_mask(mask, out_img, color)
            static_mask[mask == 255] = label  # Number indicating class

        if(viz):
            cv2.imshow("Classes", out_img)
            cv2.imshow('flow_img', flow_over_img)
            cv2.imshow('real flow', flow_img)
            cv2.waitKey(1)

        centers += centers_px  # Concat to list
        if(len(centers) > 0):
            centers = np.stack(centers)
        else:
            centers = np.zeros((0, 2))
        save_dict[im_id] = {
            "frame": img,
            "mask": static_mask,
            "centers": centers,
            "directions": directions,
            "viz_out": out_img,
            "viz_dir": flow_over_img}
    return save_dict


def collect_dataset_close_open(cfg):
    global pixel_indices
    img_size = cfg.img_size
    mask_on_close = cfg.mask_on_close
    pixel_indices = np.indices((img_size, img_size),
                               dtype=np.float32).transpose(1, 2, 0)
    # Episodes info
    # ep_lens = np.load(os.path.join(cfg.play_data_dir, "ep_lens.npy"))
    ep_start_end_ids = np.load(os.path.join(
        cfg.play_data_dir,
        "ep_start_end_ids.npy"))
    end_ids = ep_start_end_ids[:, -1]

    save_static, save_gripper = {}, {}
    # Instantiate camera to get projection and view matrices
    static_cam = hydra.utils.instantiate(
        cfg.env.cameras[0],
        cid=0, robot_id=None, objects=None)

    # Set properties needed to compute proj. matrix
    # fov=self.fov, aspect=self.aspect,
    # nearVal=self.nearval, farVal=self.farval
    gripper_cam_properties = create_gripper_cam_properties(cfg.env.cameras[1])

    # Iterate rendered_data
    files = get_files(cfg.play_data_dir, "npz")  # Sorted files
    static_hist, gripper_hist, fixed_points = [], [], []
    past_action = 1
    frame_idx = 0
    episode = 0

    # Will segment 45 frames
    back_frames_range = [50, 5]
    occ_frames_range = [0, 0]

    # Multiclass
    k_means, colors, n_classes = load_clustering(cfg.output_dir)
    start_pt = None
    for idx, filename in enumerate(tqdm.tqdm(files)):
        data = check_file(filename)
        if(data is None):
            continue  # Skip file

        _, tail = os.path.split(filename)

        # Initialize img, mask, id
        img_id = tail[:-4]
        robot_obs = data['robot_obs'][:6]  # 3 pos, 3 euler angle
        static_hist.append(
            (frame_idx, "static_%s" % img_id,
             data['rgb_static'][:, :, ::-1]))
        gripper_hist.append(
            (frame_idx, "gripper_%s" % img_id,
             robot_obs,
             data['rgb_gripper'][:, :, ::-1]))
        frame_idx += 1

        point = robot_obs[:3]

        # Start of interaction
        ep_id = int(tail[:-4].split('_')[-1])
        end_of_ep = ep_id >= end_ids[0] + 1 and len(end_ids) > 1
        if(data['actions'][-1] == 0 or end_of_ep):  # closed gripper
            # Get mask for static images
            # open -> closed
            if(past_action == 1):
                # Save static cam masks
                start_pt = point
                # save_gripper = label_gripper(gripper_cam_properties,
                #                              gripper_hist, point,
                #                              cfg.viz, save_gripper,
                #                              (img_size, img_size))

                # If region was already labeled, delete previous point
                fixed_points = update_fixed_points(
                        fixed_points,
                        point,
                        frame_idx)

                static_hist, gripper_hist = [], []
                # Occlusion back_frames_min before interaction
                occ_frames_range = [frame_idx - back_frames_range[0],
                                    frame_idx]  # min, max
            # else:
            #     mask on close
            #     Was closed and remained closed
            #     Last element in gripper_hist is the newest
            #     if(mask_on_close):
            #         save_gripper = label_gripper(gripper_cam_properties,
            #                                      [gripper_hist[-1]], point,
            #                                      cfg.viz, save_gripper,
            #                                      (img_size, img_size))
        else:  # Open gripper
            # Closed -> open transition
            occ_frames_range[1] = frame_idx
            if(past_action == 0):
                direction = point - start_pt
                direction = direction / np.linalg.norm(direction)
                class_label = k_means.predict(np.expand_dims(direction, 0)) + 1
                fixed_points.append((frame_idx, point, class_label))
                save_static = label_static(static_cam, static_hist,
                                           back_frames_range[1],
                                           occ_frames_range,
                                           n_classes, colors,
                                           fixed_points, point,
                                           cfg.viz, save_static,
                                           (img_size, img_size))

        # Reset everything
        if(end_of_ep):
            end_ids = end_ids[1:]
            fixed_points = []
            past_action = 1  # Open
            save_static, save_gripper = {}, {}
            save_data(save_static,
                      cfg.output_dir + "episode_%d" % episode,
                      sub_dir="static_cam")
            save_data(save_gripper,
                      cfg.output_dir + "episode_%d" % episode,
                      sub_dir="gripper_cam")
            episode += 1

        if (len(save_gripper.keys()) + len(save_static.keys()) > 150):
            save_data(save_static,
                      cfg.output_dir + "episode_%d" % episode,
                      sub_dir="static_cam")
            save_data(save_gripper,
                      cfg.output_dir + "episode_%d" % episode,
                      sub_dir="gripper_cam")
            save_static, save_gripper = {}, {}
        past_action = data['actions'][-1]

    save_data(save_static,
              cfg.output_dir + "episode_%d" % episode,
              sub_dir="static_cam")
    save_data(save_gripper,
              cfg.output_dir + "episode_%d" % episode,
              sub_dir="gripper_cam")
    create_data_ep_split(cfg.output_dir)


@hydra.main(config_path="../config", config_name="cfg_datacollection")
def main(cfg):
    # create_data_ep_split(cfg.output_dir)
    collect_dataset_close_open(cfg)
    # data_lst = ["%s/datasets/tabletop_directions_200px_MoC/" % cfg.project_path,
    #             "%s/datasets/vrenv_directions_200px/" % cfg.project_path]
    # merge_datasets(data_lst, cfg.output_dir)


if __name__ == "__main__":
    main()
