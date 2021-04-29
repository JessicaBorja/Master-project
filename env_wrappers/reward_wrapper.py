import gym
import numpy as np
import cv2
import pybullet as p
from utils.cam_projections import pixel2world
from utils.img_utils import visualize_np
from sklearn.cluster import DBSCAN


# As it wraps the environment, inherits its attributes
class RewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super(RewardWrapper, self).__init__(env)
        self.env = env
        self.gripper_id, self.static_id = self.find_cam_ids()
        if(self.affordance.gripper_cam.use
           and self.affordance.gripper_cam.densify_reward):
            print("Using gripper cam to shape reward")

    def find_cam_ids(self):
        gripper_id, static_id = None, None
        for i, cam in enumerate(self.cameras):
            if "gripper" in cam.name:
                gripper_id = i
            else:
                static_id = i
        return gripper_id, static_id

    def find_target_center(self, cam_id, img_obs, mask, depth):
        """
        Args:
            img_obs: np array, grayscale
                     shape = (1, img_size, img_size)
                     range = -1 to 1
            mask: np array, int64
                  shape = (1, img_size, img_size)
                  range = 0 to 1
        return:
            centers: list of 3d points (x, y, z)
        """
        # Compute affordance from static camera
        cam = self.cameras[cam_id]
        mask = np.transpose(mask, (1, 2, 0))  # (img_size, img_size, 1)

        # Visualization
        # viz_img = np.transpose(img_obs, (1, 2, 0))
        # viz_img = cv2.normalize(viz_img, None, 255, 0,
        #                         cv2.NORM_MINMAX, cv2.CV_8UC1)
        # viz_img = cv2.cvtColor(viz_img, cv2.COLOR_GRAY2RGB)
        # out_img = visualize_np(mask*255.0, viz_img, False, k=5)

        # Make clusters
        dbscan = DBSCAN(eps=3, min_samples=3)
        positives = np.argwhere(mask > 0.5)
        if(positives.shape[0] > 0):
            labels = dbscan.fit_predict(positives)
        else:
            return []

        # Find approximate center
        centers = []
        for idx, c in enumerate(np.unique(labels)):
            mid_point = np.mean(positives[np.argwhere(labels == c)], 0)[0]
            mid_point = mid_point.astype('uint8')
            u, v = mid_point[1], mid_point[0]
            # Unprojection
            world_pt = pixel2world(cam, u, v, depth)
            centers.append(world_pt)

            # Viz
            # out_img = cv2.drawMarker(out_img, (u, v),
            #                 (0, 0, 0),
            #                 markerType=cv2.MARKER_CROSS,
            #                 markerSize=5,
            #                 line_type=cv2.LINE_AA)

        # Viz imgs
        # cv2.imshow("depth", depth)
        # cv2.imshow("clusters", out_img)
        # cv2.waitKey(1)
        return centers

    def reward(self, rew):
        # modify rew
        if(self.task == "banana_combined"
           and self.affordance.gripper_cam.use
           and self.affordance.gripper_cam.densify_reward):
            obs_dict = self.get_obs()
            gripper_depth = obs_dict["depth_obs"][self.gripper_id]

            obs = self.get_gripper_obs(obs_dict)
            gripper_aff = obs["gripper_aff"]
            gripper_img = obs["gripper_img_obs"]
            centers = self.find_target_center(self.gripper_id,
                                              gripper_img,
                                              gripper_aff,
                                              gripper_depth)
            tcp_pos = obs_dict["robot_obs"][:3]
            # Maximum distance given the task
            closest_pt_dist = self.banana_radio
            pt = None
            for c in centers:
                distance = np.linalg.norm(tcp_pos - c)
                if(distance < closest_pt_dist):
                    closest_pt_dist = distance
                    pt = c
            # See selected point
            # if(pt is not None):
            #     p.addUserDebugText("gripper pt",
            #                        textPosition=pt, textColorRGB=[1, 0, 0])
            # Create positive reward relative to the distance
            # between the closest point detected by the affordances
            # and the end effector position
            rew = rew + (1 - (closest_pt_dist / self.banana_radio))
        return rew
