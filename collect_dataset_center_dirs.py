import hydra
import utils.flowlib as flowlib
from utils.img_utils import overlay_mask, tresh_np, overlay_flow
from utils.label_segmentation import get_static_mask, get_gripper_mask
from utils.file_manipulation import get_files, save_data, create_data_ep_split
import cv2
import numpy as np
import tqdm
import os
import sys
parent_dir = os.path.dirname(os.getcwd())
sys.path.insert(0, parent_dir)
sys.path.insert(0, parent_dir+"/VREnv/")


# Keep points in a distance larger than radius from new_point
# Do not keep fixed points more than 100 frames
def update_fixed_points(fixed_points, new_point,
                        current_frame_idx, radius=0.15):
    x = []
    for frame_idx, p in fixed_points:
        if np.linalg.norm(new_point - p) > radius:
            # and current_frame_idx - frame_idx < 1500:
            x.append((frame_idx, p))
    # x = [ p for (frame_idx, p) in fixed_points if
    # ( np.linalg.norm(new_point - p) > radius)]
    # # and current_frame_idx - frame_idx < 100 )
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


def update_mask(fixed_points, mask, directions,
                frame_img_tuple, cam, out_img_size):
    # Update masks with fixed_points
    centers = []
    (frame_timestep, img) = frame_img_tuple
    for point_timestep, p in fixed_points:
        # Only add point if it was fixed before seing img
        if(frame_timestep >= point_timestep):
            new_mask, center_px = get_static_mask(cam, img, p)
            new_mask, center_px = resize_mask_and_center(new_mask, center_px,
                                                         out_img_size)
            centers.append(center_px)
            directions = label_directions(center_px, new_mask, directions)
            mask = overlay_mask(new_mask, mask, (255, 255, 255))
    return mask, centers, directions


def check_file(filename, allow_pickle=True):
    try:
        data = np.load(filename, allow_pickle=allow_pickle)
        if(len(data['rgb_static'].shape) != 3 or
                len(data['rgb_gripper'].shape) != 3):
            raise Exception("Corrupt data")
    except Exception as e:
        # print(e)
        data = None
    return data


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
                cv2.imshow('Gripperreal flow', flow_img)
                cv2.waitKey(1)

            save_dict[im_id] = {
                "frame": img,
                "mask": mask,
                "centers": np.stack([center_px]),
                "directions": directions,
                "viz_out": out_img}
    return save_dict


def resize_mask_and_center(mask, center, new_size):
    orig_H, orig_W = mask.shape[:2]
    mask = cv2.resize(mask, new_size)
    center = np.array(center) * new_size[0] // orig_H
    return mask, center


def label_static(static_cam, static_hist, back_min, back_max,
                 fixed_points, pt, viz, save_dict, out_img_size):
    for idx, (fr_idx, im_id, img) in enumerate(static_hist):
        # For static mask assume oclusion
        # until back_frames_min before
        centers = []
        H, W = out_img_size  # img.shape[:2]  # img_shape = (H, W, C)
        directions = np.stack([np.ones((H, W)),
                               np.zeros((H, W))], axis=-1).astype(np.float32)
        fp_mask, centers_px, fp_directions = update_mask(
            fixed_points,
            np.zeros((H, W)),
            directions,
            (fr_idx, img),
            static_cam,
            out_img_size)
        # first create fp masks and place current(newest)
        # mask and optical flow on top
        if(idx <= len(static_hist) - back_min and
                idx > len(static_hist) - back_max):
            # Get new grip
            mask, center_px = get_static_mask(static_cam, img, pt)
            mask, center_px = resize_mask_and_center(mask, center_px,
                                                     out_img_size)
            directions = label_directions(center_px, mask, fp_directions)
            centers.append(center_px)
        else:
            # No segmentation in current image due to occlusion
            mask = np.zeros((H, W))

        # Visualize
        img = cv2.resize(img, out_img_size)
        out_separate = overlay_mask(
            fp_mask,
            img,
            (255, 0, 0))
        out_separate = overlay_mask(
            mask,
            out_separate,
            (0, 0, 255))

        # Real mask
        static_mask = overlay_mask(mask, fp_mask, (255, 255, 255))
        out_img = overlay_mask(static_mask, img, (0, 0, 255))
        flow_img = flowlib.flow_to_image(directions)[:, :, ::-1]
        flow_over_img = overlay_flow(flow_img, img, static_mask)
        if(viz):
            cv2.imshow("Separate", out_separate)
            cv2.imshow("Real", out_img)
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
            "viz_out": out_separate}
    return save_dict


def collect_dataset_close_open(cfg):
    global pixel_indices
    img_size = cfg.img_size
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
    # Will segment 40 frames
    back_frames_max = 50
    back_frames_min = 5
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
                save_static = label_static(static_cam, static_hist,
                                           back_frames_min, back_frames_max,
                                           fixed_points, point,
                                           cfg.viz, save_static,
                                           (img_size, img_size))
                save_gripper = label_gripper(gripper_cam_properties,
                                             gripper_hist, point,
                                             cfg.viz, save_gripper,
                                             (img_size, img_size))

                # If region was already labeled, delete previous point
                fixed_points = update_fixed_points(
                        fixed_points,
                        point,
                        frame_idx)

                static_hist, gripper_hist = [], []
            # else:  # mask on close
            #     # Was closed and remained closed
            #     # Last element in gripper_hist is the newest
            #     save_gripper = label_gripper(gripper_cam_properties,
            #                                  [gripper_hist[-1]], point,
            #                                  cfg.viz, save_gripper)
        # Open gripper
        else:
            # Closed -> open transition
            if(past_action == 0):
                curr_point = data['robot_obs'][:3]
                fixed_points.append((frame_idx, curr_point))

        # Reset everything
        if(end_of_ep):
            end_ids = end_ids[1:]
            fixed_points = []
            past_action = 1  # Open
            save_static, save_gripper = {}, {}
            save_data(save_static,
                      cfg.save_dir + "episode_%d" % episode,
                      sub_dir="static_cam")
            save_data(save_gripper,
                      cfg.save_dir + "episode_%d" % episode,
                      sub_dir="gripper_cam")
            episode += 1

        if (len(save_gripper.keys()) + len(save_static.keys()) > 150):
            save_data(save_static,
                      cfg.save_dir + "episode_%d" % episode,
                      sub_dir="static_cam")
            save_data(save_gripper,
                      cfg.save_dir + "episode_%d" % episode,
                      sub_dir="gripper_cam")
            save_static, save_gripper = {}, {}
        past_action = data['actions'][-1]

    save_data(save_static,
              cfg.save_dir + "episode_%d" % episode,
              sub_dir="static_cam")
    save_data(save_gripper,
              cfg.save_dir + "episode_%d" % episode,
              sub_dir="gripper_cam")
    create_data_ep_split(cfg.save_dir)


@hydra.main(config_path="./config", config_name="cfg_datacollection")
def main(cfg):
    collect_dataset_close_open(cfg)
    # create_data_ep_split(cfg.save_dir)


if __name__ == "__main__":
    main()
