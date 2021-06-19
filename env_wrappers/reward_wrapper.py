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
from utils.img_utils import overlay_mask


# As it wraps the environment, inherits its attributes
class RewardWrapper(gym.RewardWrapper):
    def __init__(self, env, max_ts=100):
        super(RewardWrapper, self).__init__(env)
        self.env = env
        self.gripper_id, self.static_id = self.find_cam_ids()
        self.ts_counter = 0
        self.max_ts = max_ts
        if(self.affordance.gripper_cam.densify_reward):
            print("RewardWrapper: Gripper cam to shape reward")

    def find_cam_ids(self):
        gripper_id, static_id = None, None
        for i, cam in enumerate(self.cameras):
            if "gripper" in cam.name:
                gripper_id = i
            else:
                static_id = i
        return gripper_id, static_id

    # Clustering
    def _clusters_find_target_center(self, cam_id, img_obs, depth, obs):
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
        # cv2.imshow("clusters", out_img)
        # cv2.waitKey(1)
        return cluster_outputs

    def reward(self, rew):
        # modify rew
        if(self.task == "banana_combined" or self.task == "pickup"
           and self.affordance.gripper_cam.densify_reward):
            # set by observation wrapper so that
            # both have the same observation on
            # a given timestep
            if(self.env.curr_raw_obs is not None):
                obs_dict = self.env.curr_raw_obs
            else:
                obs_dict = self.get_obs()
            tcp_pos = obs_dict["robot_obs"][:3]

            # Create positive reward relative to the distance
            # between the closest point detected by the affordances
            # and the end effector position
            # p.addUserDebugText("target_reward",
            #                    textPosition=self.unwrapped.current_target,
            #                    textColorRGB=[0, 1, 0])

            # If episode is not done because of moving to far away
            if(not self.env.unwrapped._termination()
               and self.ts_counter <= self.max_ts):
                distance = np.linalg.norm(tcp_pos - self.unwrapped.current_target)
                # cannot be larger than 1
                scale_dist = min(distance / self.target_radius, 1)
                rew += (1 - scale_dist)**0.4
                self.ts_counter += 1
            else:
                # If episode was successful
                if(rew >= 1):
                    rew += self.max_ts - self.ts_counter
                self.ts_counter = 0
        return rew
