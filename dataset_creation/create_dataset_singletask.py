import hydra
import cv2
import numpy as np
import tqdm
import os
from omegaconf import OmegaConf
import glob
import vapo.utils.flowlib as flowlib
from vapo.utils.img_utils import overlay_mask, tresh_np, overlay_flow
from vapo.utils.label_segmentation import get_static_mask, get_gripper_mask
from vapo.utils.file_manipulation import get_files, save_data, check_file,\
                                    create_data_ep_split
from dataset_creation.cameras.real_cameras import CamProjections


# Keep points in a distance larger than radius from new_point
# Do not keep fixed points more than 100 frames
def update_fixed_points(fixed_points, new_point,
                        current_frame_idx, radius=0.08):
    x = []
    for frame_idx, p in fixed_points:
        # Point not in air
        if (np.linalg.norm(new_point - p) > radius):
            # and current_frame_idx - frame_idx < 1500:
            x.append((frame_idx, p))
    # x = [ p for (frame_idx, p) in fixed_points if
    # ( np.linalg.norm(new_point - p) > radius)]
    # # and current_frame_idx - frame_idx < 100 )
    return x


def label_directions(center, object_mask, direction_labels, camtype):
    # Get directions
    # Shape: [H x W x 2]
    indices = pixel_indices[camtype]
    object_mask = tresh_np(object_mask, 100)
    object_center_directions = (center - indices).astype(np.float32)
    object_center_directions = object_center_directions\
        / np.maximum(np.linalg.norm(object_center_directions,
                                    axis=2, keepdims=True), 1e-10)

    # Add it to the labels
    direction_labels[object_mask == 1] = \
        object_center_directions[object_mask == 1]
    return direction_labels


def update_mask(fixed_points, mask, directions,
                frame_img_tuple, cam, out_img_size,
                teleop_data=False, radius=10):
    # Update masks with fixed_points
    centers = []
    (frame_timestep, img) = frame_img_tuple
    for point_timestep, p in fixed_points:
        # Only add point if it was fixed before seing img
        if(frame_timestep >= point_timestep):
            new_mask, center_px = get_static_mask(cam, img, p, r=radius,
                                                  teleop_data=teleop_data)
            new_mask, center_px = resize_mask_and_center(new_mask, center_px,
                                                         out_img_size)
            centers.append(center_px)
            directions = label_directions(center_px, new_mask,
                                          directions, "static")
            mask = overlay_mask(new_mask, mask, (255, 255, 255))
    return mask, centers, directions


def label_gripper(cam, img_hist, back_frames_max, curr_pt,
                  last_pt, viz, save_dict, out_img_size,
                  closed_gripper=False, teleop_data=False,
                  radius=25):
    for idx, (fr_idx, im_id, robot_obs, img) in enumerate(img_hist):
        if(im_id not in save_dict):
            # Shape: [H x W x 2]
            H, W = out_img_size  # img.shape[:2]
            directions = np.stack(
                            [np.ones((H, W)),
                             np.zeros((H, W))], axis=-1).astype(np.float32)
            mask = np.zeros(out_img_size)
            if(robot_obs[-1] > 0.015):
                for point in [curr_pt, last_pt]:
                    if point is not None:
                        # Center and directions in matrix convention (row, column)
                        new_mask, center_px = get_gripper_mask(img,
                                                               robot_obs[:-1],
                                                               point,
                                                               cam,
                                                               radius=radius,
                                                               teleop_data=teleop_data)
                        new_mask, center_px = resize_mask_and_center(new_mask,
                                                                     center_px,
                                                                     out_img_size)
                        if(np.any(center_px < 0) or np.any(center_px >= H)):
                            new_mask = np.zeros(out_img_size)
                            center_px = np.array([1, 0])
                            # continue  # Outside of image FOV
                        directions = label_directions(center_px,
                                                      new_mask,
                                                      directions,
                                                      "gripper")
                        mask = overlay_mask(new_mask, mask, (255, 255, 255))
            else:
                mask = np.zeros(out_img_size)
                center_px = np.array([1, 0])

            # Visualize results
            img = cv2.resize(img, out_img_size)
            out_img = overlay_mask(mask, img, (0, 0, 255))
            flow_img = flowlib.flow_to_image(directions)[:, :, ::-1]
            flow_over_img = overlay_flow(flow_img, img, mask)

            if(viz):
                viz_img = cv2.resize(out_img, (200, 200))
                viz_flow = cv2.resize(flow_img, (200, 200))
                viz_flow_over_img = cv2.resize(flow_over_img, (200, 200))
                cv2.imshow("Gripper", viz_img)
                cv2.imshow('Gripper flow_img', viz_flow_over_img)
                cv2.imshow('Gripper real flow', viz_flow)
                cv2.waitKey(1)

            save_dict[im_id] = {
                "frame": img,
                "mask": mask,
                "centers": np.stack([center_px]),
                "directions": directions,
                "viz_out": out_img,
                "viz_dir": flow_over_img,
                "gripper_width": robot_obs[-1]}
    return save_dict


def resize_mask_and_center(mask, center, new_size):
    orig_H, orig_W = mask.shape[:2]
    mask = cv2.resize(mask, new_size)
    center = np.array(center) * new_size[0] // orig_H
    return mask, center


def label_static(static_cam, static_hist, back_min, back_max,
                 fixed_points, pt, viz, save_dict, out_img_size,
                 teleop_data=False, radius=10):
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
            out_img_size,
            teleop_data=teleop_data,
            radius=radius)
        # first create fp masks and place current(newest)
        # mask and optical flow on top
        if(idx <= len(static_hist) - back_min and
                idx > len(static_hist) - back_max):
            # Get new grip
            mask, center_px = get_static_mask(static_cam, img, pt, r=radius,
                                              teleop_data=teleop_data)
            mask, center_px = resize_mask_and_center(mask, center_px,
                                                     out_img_size)
            directions = label_directions(center_px, mask,
                                          fp_directions, "static")
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
            "viz_out": out_img,
            "viz_dir": flow_over_img}
    return save_dict


def instantiate_cameras(cfg, teleop_data):
    if(not teleop_data):
        # Instantiate camera to get projection and view matrices
        static_cam = hydra.utils.instantiate(
            cfg.env.cameras[0],
            cid=0, robot_id=None, objects=None)

        # Set properties needed to compute proj. matrix
        # fov=self.fov, aspect=self.aspect,
        # nearVal=self.nearval, farVal=self.farval
        gripper_cam = hydra.utils.instantiate(
            cfg.env.cameras[1],
            cid=1, robot_id=None, objects=None)
    else:
        cam_params_path = cfg.play_data_dir
        dir_content = glob.glob(cfg.play_data_dir)
        if("camera_info.npz" in dir_content):
            cam_info = np.load(os.path.join(
                               cfg.play_data_dir,
                               "camera_info.npz"),
                               allow_pickle=True)
        else:
            # Has subfolders of recorded data
            cam_params_path = glob.glob(cfg.play_data_dir + "/*/")[0]
            cam_info = np.load(os.path.join(
                    cam_params_path,
                    "camera_info.npz"),
                    allow_pickle=True)
        teleop_cfg = OmegaConf.load(os.path.join(
                                        cam_params_path,
                                        ".hydra/config.yaml"))
        gripper_cfg = teleop_cfg.cams.gripper_cam
        gripper_cam = CamProjections(
                             cam_info["gripper_intrinsics"].item(),
                             cam_info["gripper_extrinsic_calibration"],
                             resize_resolution=gripper_cfg.resize_resolution,
                             crop_coords=gripper_cfg.crop_coords,
                             resolution=gripper_cfg.resolution,
                             name=gripper_cfg.name)
        static_cfg = teleop_cfg.cams.static_cam
        static_cam = CamProjections(
                            cam_info["static_intrinsics"].item(),
                            cam_info['static_extrinsic_calibration'],
                            resize_resolution=static_cfg.resize_resolution,
                            crop_coords=static_cfg.crop_coords,
                            resolution=static_cfg.resolution,
                            name="static")
    return static_cam, gripper_cam


def collect_dataset_close_open(cfg):
    global pixel_indices
    save_viz = cfg.save_viz
    gripper_out_size = (cfg.img_size.gripper, cfg.img_size.gripper)
    static_out_size = (cfg.img_size.static, cfg.img_size.static)
    pixel_indices = {"gripper": np.indices(gripper_out_size,
                                           dtype=np.float32).transpose(1, 2, 0),
                     "static": np.indices(static_out_size,
                                          dtype=np.float32).transpose(1, 2, 0)}
    # Hyperparameters
    mask_on_close = cfg.mask_on_close
    back_frames_max = cfg.labeling.back_frames_max
    back_frames_min = cfg.labeling.back_frames_min
    static_r = cfg.labeling.label_size.static_cam
    gripper_r = cfg.labeling.label_size.gripper_cam
    teleop_data = cfg.labeling.teleop_data
    fixed_pt_radius = cfg.labeling.fixed_pt_radius

    # Episodes info
    # Sorted files
    files = []
    if(not teleop_data):
        # Simulation
        # ep_lens = np.load(os.path.join(cfg.play_data_dir, "ep_lens.npy"))
        ep_start_end_ids = np.load(os.path.join(
            cfg.play_data_dir,
            "ep_start_end_ids.npy"))
        end_ids = ep_start_end_ids[:, -1]
        files = get_files(cfg.play_data_dir, "npz")
    else:
        # Real life experiments
        # Play data dir contains subdirectories
        # With different data collection runs
        episodes = glob.glob(cfg.play_data_dir + '/*/')
        episodes = {ep_path: len(glob.glob(ep_path + '*.npz')) - 2
                for  ep_path in episodes}
        end_ids = list(episodes.values())
        for ep_path in episodes.keys():
            f = get_files(ep_path, "npz")
            f.remove(os.path.join(ep_path, "camera_info.npz"))
            files.extend(f)
    save_static, save_gripper = {}, {}
    static_cam, gripper_cam = instantiate_cameras(cfg, teleop_data)

    static_hist, gripper_hist, fixed_points = [], [], []
    past_action = 1
    frame_idx = 0
    episode = 0
    last_pt = None

    # Iterate rendered_data
    head, tail = os.path.split(files[0])
    _,  curr_folder = os.path.split(head)
    for idx, filename in enumerate(tqdm.tqdm(files)):
        data = check_file(filename)
        if(data is None):
            continue  # Skip file
        if('robot_state' not in data):
            continue

        if(idx < len(files) - 1):
            next_folder = os.path.split(
                        os.path.split(files[idx + 1])[0])[-1]
        else:
            next_folder = curr_folder

        # Initialize img, mask, id
        head, tail = os.path.split(filename)
        img_id = tail[:-4]
        if(teleop_data):
            proprio = data["robot_state"].item()
            # orn = p.getEulerFromQuaternion(proprio["tcp_orn"])
            orn = proprio["tcp_orn"]
            robot_obs = np.array([*proprio["tcp_pos"],
                                  *orn,
                                  proprio["gripper_opening_width"]])
        else:
            robot_obs = data["robot_obs"][:7]  # 3 pos, 3 euler angle
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
        if(not teleop_data):
            ep_id = int(tail[:-4].split('_')[-1])
            end_of_ep = ep_id >= end_ids[0] + 1 and len(end_ids) > 1
            gripper_action = data['actions'][-1]  # -1 -> closed, 1 -> open
        else:
            ep_id = int(tail[:-4].split('_')[-1])
            end_of_ep = (ep_id >= end_ids[0] and len(end_ids) > 1)\
                or curr_folder != next_folder
            gripper_action = robot_obs[-1] > 0.077  # Open
            gripper_action = (data['action'].item()['motion'][-1] + 1)/2
        if(gripper_action <= 0 or end_of_ep):  # closed gripper
            # Get mask for static images
            # open -> closed
            if(past_action == 1):
                # Save static cam masks
                save_static = label_static(static_cam, static_hist,
                                           back_frames_min, back_frames_max,
                                           fixed_points, point,
                                           cfg.viz, save_static,
                                           static_out_size,
                                           teleop_data=teleop_data,
                                           radius=static_r)
                save_gripper = label_gripper(gripper_cam,
                                             gripper_hist,
                                             back_frames_max,
                                             point, last_pt,
                                             cfg.viz, save_gripper,
                                             gripper_out_size,
                                             teleop_data=teleop_data,
                                             radius=gripper_r)
                # If region was already labeled, delete previous point
                fixed_points = update_fixed_points(
                        fixed_points,
                        point,
                        frame_idx,
                        radius=fixed_pt_radius)

                static_hist, gripper_hist = [], []
            else:
                # mask on close
                # Was closed and remained closed
                # Last element in gripper_hist is the newest
                if(mask_on_close):
                    save_gripper = label_gripper(gripper_cam,
                                                 [gripper_hist[-1]],
                                                 back_frames_max,
                                                 point, last_pt,
                                                 cfg.viz, save_gripper,
                                                 gripper_out_size,
                                                 closed_gripper=True,
                                                 teleop_data=teleop_data,
                                                 radius=gripper_r)
                    gripper_hist = gripper_hist[:-1]
        # Open gripper
        else:
            # Closed -> open transition
            # and point not in air..
            if(past_action <= 0 and
               point[-1] <= 0.21):
                curr_point = point
                fixed_points.append((frame_idx, curr_point))
                last_pt = curr_point

        # Reset everything
        if(end_of_ep):
            end_ids = end_ids[1:]
            fixed_points = []
            past_action = 1  # Open
            curr_folder = next_folder
            save_data(save_static,
                      cfg.output_dir + "episode_%02d" % episode,
                      sub_dir="static_cam",
                      save_viz=save_viz)
            save_data(save_gripper,
                      cfg.output_dir + "episode_%02d" % episode,
                      sub_dir="gripper_cam",
                      save_viz=save_viz)
            save_static, save_gripper = {}, {}
            episode += 1

        if (len(save_gripper.keys()) + len(save_static.keys()) > 150):
            save_data(save_static,
                      cfg.output_dir + "episode_%02d" % episode,
                      sub_dir="static_cam",
                      save_viz=save_viz)
            save_data(save_gripper,
                      cfg.output_dir + "episode_%02d" % episode,
                      sub_dir="gripper_cam",
                      save_viz=save_viz)
            save_static, save_gripper = {}, {}
        past_action = gripper_action

    save_data(save_static,
              cfg.output_dir + "episode_%02d" % episode,
              sub_dir="static_cam",
              save_viz=save_viz)
    save_data(save_gripper,
              cfg.output_dir + "episode_%02d" % episode,
              sub_dir="gripper_cam",
              save_viz=save_viz)
    create_data_ep_split(cfg.output_dir, cfg.labeling.split_by_episodes)


@hydra.main(config_path="../config", config_name="cfg_datacollection")
def main(cfg):
    collect_dataset_close_open(cfg)
    # data_lst = ["%s/datasets/real_world/ep_%d"%(cfg.project_path, i) for i in range(1, 8)]
    # data_lst = ["%s/datasets/teleop_real/teleop_real_08_09/" % cfg.project_path,
    #             "%s/datasets/teleop_real/teleop_real_01_09/" % cfg.project_path]
    # merge_datasets(data_lst, cfg.output_dir)
    # create_data_ep_split(cfg.output_dir, cfg.labeling.split_by_episodes)

if __name__ == "__main__":
    main()
