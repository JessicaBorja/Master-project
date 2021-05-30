import gym
import numpy as np
import cv2
import pybullet as p
import torch
from utils.cam_projections import pixel2world
from utils.img_utils import overlay_mask
from sklearn.cluster import DBSCAN
from matplotlib import cm

from sac_agent.sac_utils.utils import tt
from utils.cam_projections import pixel2world
from utils.img_utils import torch_to_numpy, overlay_mask, viz_aff_centers_preds


# As it wraps the environment, inherits its attributes
class RewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super(RewardWrapper, self).__init__(env)
        self.env = env
        self.gripper_id, self.static_id = self.find_cam_ids()
        if(self.affordance.gripper_cam.use
           and self.affordance.gripper_cam.densify_reward):
            print("RewardWrapper: Gripper cam to shape reward")
        # Combined model should initialize this (self.env.current_target)
        self.current_target = None

    def find_cam_ids(self):
        gripper_id, static_id = None, None
        for i, cam in enumerate(self.cameras):
            if "gripper" in cam.name:
                gripper_id = i
            else:
                static_id = i
        return gripper_id, static_id

    # Clustering
    def _find_target_center(self, cam_id, img_obs, depth, obs):
        """
        Args:
            img_obs: np array, RGB original resolution from camera
                     shape = (1, cam.height, cam.width)
                     range = 0 to 255
                     Only used for vizualization purposes
            obs: dict()
                 "img_obs": img_obs after transforms
                 "gripper_aff": afffordance mask for the gripper cam
                        np array, int64
                        shape = (1, img_size, img_size)
                        range = 0 to 1
        return:
            centers: list of 3d points (x, y, z)
        """
        mask = obs["gripper_aff"]

        # Compute affordance from camera
        cam = self.cameras[cam_id]
        mask = np.transpose(mask, (1, 2, 0))  # (img_size, img_size, 1)

        # Scale affordance mask to camera W,H rendering
        mask_scaled = cv2.resize((mask*255).astype('uint8'),  # To keep in-between values
                                 dsize=(cam.height, cam.width),
                                 interpolation=cv2.INTER_CUBIC) / 255.0  # to scale them back between 0-1

        # Make clusters
        dbscan = DBSCAN(eps=3, min_samples=3)
        positives = np.argwhere(mask_scaled > 0.3)
        cluster_outputs = []
        if(positives.shape[0] > 0):
            labels = dbscan.fit_predict(positives)
        else:
            return cluster_outputs

        cluster_ids = np.unique(labels)

        # Visualization
        # Uses openCV which needs BGR
        # out_img = visualize_np(mask_scaled*255.0, img_obs[:, :, ::-1])

        # Create cluster segmentation mask
        # colors = cm.jet(np.linspace(0, 1, len(cluster_ids)))
        out_img = overlay_mask(mask_scaled * 255.0,
                               img_obs[:, :, ::-1],
                               (0, 0, 255))

        # Find approximate center
        n_pixels = mask_scaled.shape[0] * mask_scaled.shape[1]
        for idx, c in enumerate(cluster_ids):
            cluster = positives[np.argwhere(labels == c).squeeze()]  # N, 3
            if(len(cluster.shape) == 1):
                cluster = np.expand_dims(cluster, 0)
            pixel_count = cluster.shape[0] / n_pixels

            # # Visualize all cluster
            # out_img[cluster[:, 0], cluster[:, 1]] = \
            #         [int(a * 255) for a in colors[idx][:3]]

            if(pixel_count < 0.02):  # Skip small clusters
                continue

            # img size is 300 then we need int64 to cover all pixels
            mid_point = np.mean(cluster, 0).astype('int')
            u, v = mid_point[1], mid_point[0]
            # Unprojection
            world_pt = pixel2world(cam, u, v, depth)
            robustness = np.mean(mask_scaled[cluster])
            c_out = {"center": world_pt,
                     "pixel_count": pixel_count,
                     "robustness": robustness}
            cluster_outputs.append(c_out)

            # Viz
            out_img = cv2.drawMarker(out_img, (u, v),
                                     (0, 0, 0),
                                     markerType=cv2.MARKER_CROSS,
                                     markerSize=5,
                                     line_type=cv2.LINE_AA)

        # Viz imgs
        # cv2.imshow("depth", depth)
        cv2.imshow("clusters", out_img)
        cv2.waitKey(1)
        return cluster_outputs

    # Aff-center 
    def find_target_center(self, cam_id, orig_img, depth, obs):
        """
        Args:
            orig_img: np array, RGB original resolution from camera
                     shape = (1, cam.height, cam.width)
                     range = 0 to 255
                     Only used for vizualization purposes
            obs: dictionary
                - "img_obs":
                - "gripper_aff": 
                    affordance segmentation mask, range 0-1
                    np.array(size=(1, img_size,img_size))
                - "gripper_aff_probs":
                    affordance activation function output
                    np.array(size=(1, n_classes, img_size,img_size))
                    range 0-1
                - "gripper_center_dir": center direction predictions
                    vectors in pixel space
                    np.array(size=(1, 2, img_size,img_size))
                np array, int64
                  shape = (1, img_size, img_size)
                  range = 0 to 1
        return:
            centers: list of 3d points (x, y, z)
        """
        aff_mask = obs["gripper_aff"]
        aff_probs = obs["gripper_aff_probs"]
        directions = obs["gripper_center_dir"]
        cam = self.cameras[cam_id]

        # Predict affordances and centers
        aff_mask, center_dir, object_centers, object_masks = \
            self.gripper_cam_aff_net.predict(tt(aff_mask), tt(directions))

        # Visualize predictions
        viz_aff_centers_preds(orig_img, aff_mask, tt(aff_probs), center_dir,
                              object_centers, object_masks)

        # Plot different objects
        cluster_outputs = []
        object_centers = [torch_to_numpy(o) for o in object_centers]
        if(len(object_centers) > 0):
            target_px = object_centers[0]
        else:
            return cluster_outputs

        # To numpy
        aff_probs = np.transpose(aff_probs[0], (1, 2, 0))  # H, W, 2
        object_masks = torch_to_numpy(object_masks[0])  # H, W

        max_robustness = 0
        obj_class = np.unique(object_masks)[1:]
        obj_class = obj_class[obj_class != 0]  # remove background class

        # Look for most likely center
        n_pixels = aff_mask.shape[1] * aff_mask.shape[2]
        for i, o in enumerate(object_centers):
            # Mean prob of being class 1 (foreground)
            cluster = aff_probs[object_masks == obj_class[i], 1]
            robustness = np.mean(cluster)
            pixel_count = cluster.shape[0] / n_pixels
            if(pixel_count < 0.02):  # Skip small clusters
                continue
            v, u = o
            world_pt = pixel2world(cam, u, v, depth)
            c_out = {"center": world_pt,
                     "pixel_count": pixel_count,
                     "robustness": max_robustness}
            cluster_outputs.append(c_out)
            if(robustness > max_robustness):
                max_robustness = robustness
                target_px = o

        # Convert back to observation size
        pred_shape = aff_probs.shape[:2]
        orig_shape = depth.shape[:2]
        target_px = (target_px * orig_shape / pred_shape).astype("int64")

        # world cord
        v, u = target_px
        out_img = cv2.drawMarker(np.array(orig_img[:, :, ::-1]),
                                 (u, v),
                                 (0, 255, 0),
                                 markerType=cv2.MARKER_CROSS,
                                 markerSize=12,
                                 line_type=cv2.LINE_AA)
        cv2.imshow("out_img", out_img)
        cv2.imshow("depth", depth)

        # Compute depth
        target_pos = pixel2world(cam, u, v, depth)
        target_pos = np.array(target_pos)
        return cluster_outputs

    def reward(self, rew):
        # modify rew
        if(self.task == "banana_combined" or self.task == "pickup"
           and self.affordance.gripper_cam.use
           and self.affordance.gripper_cam.densify_reward):
            # set by observation wrapper so that
            # both have the same observation on
            # a given timestep
            if(self.env.curr_raw_obs is not None):
                obs_dict = self.env.curr_raw_obs
            else:
                obs_dict = self.get_obs()
            # Cam resolution image
            gripper_depth = obs_dict["depth_obs"][self.gripper_id]
            gripper_img_orig = obs_dict["rgb_obs"][self.gripper_id]

            # RL resolution (64x64). What the agent observes
            # Preprocessed by obs_wrapper
            if(self.env.curr_processed_obs is not None):
                obs = self.env.curr_processed_obs
            else:
                obs = self.get_gripper_obs(obs_dict)

            # px count amount of pixels in cluster relative to
            # amount of pixels in img
            clusters_outputs = self.find_target_center(self.gripper_id,
                                                       gripper_img_orig,
                                                       gripper_depth,
                                                       obs)
            tcp_pos = obs_dict["robot_obs"][:3]

            p.removeAllUserDebugItems()
            p.addUserDebugText("target",
                               textPosition=self.current_target,
                               textColorRGB=[1, 0, 0])
            # Maximum distance given the task
            for out_dict in clusters_outputs:
                c = out_dict["center"]
                # If aff detects closer target which is large enough
                # and Detected affordance close to target
                if(np.linalg.norm(self.current_target - c) < 0.05):
                    self.current_target = c

            # See selected point
            # p.removeAllUserDebugItems()
            p.addUserDebugText("target",
                               textPosition=self.current_target,
                               textColorRGB=[1, 0, 0])

            # Create positive reward relative to the distance
            # between the closest point detected by the affordances
            # and the end effector position
            distance = np.linalg.norm(tcp_pos - self.current_target)
            if(self.env.unwrapped._termination()):
                rew = -1
            else:
                scale_dist = min(distance / self.target_radius, 1)  # cannot be larger than 1
                rew += (1 - scale_dist)**(0.4)
        return rew
