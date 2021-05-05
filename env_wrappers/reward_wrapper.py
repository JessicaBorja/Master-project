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
        self.current_target = None  # Combined model should initialize this

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
            img_obs: np array, RGB original resolution from camera
                     shape = (1, cam.height, cam.width)
                     range = 0 to 255
                     Only used for vizualization purposes
            mask: np array, int64
                  shape = (1, img_size, img_size)
                  range = 0 to 1
        return:
            centers: list of 3d points (x, y, z)
        """
        # Compute affordance from camera
        cam = self.cameras[cam_id]
        mask = np.transpose(mask, (1, 2, 0))  # (img_size, img_size, 1)

        # Scale affordance mask to camera W,H rendering
        mask_scaled = cv2.resize((mask*255).astype('uint8'),  # To keep in-between values
                                 dsize=(cam.height, cam.width),
                                 interpolation=cv2.INTER_CUBIC) / 255.0  # to scale them back between 0-1

        # Visualization
        # Uses openCV which needs BGR
        # out_img = visualize_np(mask_scaled*255.0, img_obs[:, :, ::-1])

        # Make clusters
        dbscan = DBSCAN(eps=3, min_samples=3)
        positives = np.argwhere(mask_scaled > 0.5)
        centers, pixel_counters = [], []
        if(positives.shape[0] > 0):
            labels = dbscan.fit_predict(positives)
        else:
            return (centers, pixel_counters)

        # Find approximate center
        n_pixels = mask_scaled.shape[0] * mask_scaled.shape[1]
        for idx, c in enumerate(np.unique(labels)):
            cluster = positives[np.argwhere(labels == c).squeeze()]  # N, 3
            if(len(cluster.shape) == 1):
                cluster = np.expand_dims(cluster, 0)
            pixel_count = cluster.shape[0] / n_pixels
            mid_point = np.mean(cluster, 0)[:2]
            mid_point = mid_point.astype('uint8')
            u, v = mid_point[1], mid_point[0]
            # Unprojection
            world_pt = pixel2world(cam, u, v, depth)
            # p.addUserDebugText("O", textPosition=world_pt, textColorRGB=[1, 0, 0])
            centers.append(world_pt)
            pixel_counters.append(pixel_count)

        #     # Viz
        #     out_img = cv2.drawMarker(out_img, (u, v),
        #                              (0, 0, 0),
        #                              markerType=cv2.MARKER_CROSS,
        #                              markerSize=5,
        #                              line_type=cv2.LINE_AA)

        # # Viz imgs
        # cv2.imshow("depth", depth)
        # cv2.imshow("clusters", out_img)
        # cv2.waitKey(1)
        return (centers, pixel_counters)

    def reward(self, rew):
        # modify rew
        if(self.task == "banana_combined"
           and self.affordance.gripper_cam.use
           and self.affordance.gripper_cam.densify_reward):
            obs_dict = self.get_obs()
            # Cam resolution image
            gripper_depth = obs_dict["depth_obs"][self.gripper_id]
            gripper_img_orig = obs_dict["rgb_obs"][self.gripper_id] 

            # RL resolution (64x64). What the agent observes
            # Preprocessed by obs_wrapper
            obs = self.get_gripper_obs(obs_dict)
            gripper_aff = obs["gripper_aff"]
            # gripper_img = obs["gripper_img_obs"]

            # px count amount of pixels in cluster relative to
            # amount of pixels in img
            centers, px_count = self.find_target_center(self.gripper_id,
                                                        gripper_img_orig,
                                                        gripper_aff,
                                                        gripper_depth)
            tcp_pos = obs_dict["robot_obs"][:3]
            # Maximum distance given the task
            # pt = None
            for c, n_px in zip(centers, px_count):

                # If aff detects closer target which is large enough
                # and Detected affordance close to target
                if(np.linalg.norm(self.current_target - c) < 0.05
                   and n_px > 0.02):
                    # pt = c
                    self.current_target = c

            # See selected point
            # if(pt is not None):
            #     # p.loadURDF("sphere_small.urdf", pt)
            #     p.addUserDebugText("O",
            #                        textPosition=pt, textColorRGB=[1, 0, 0])

            # Create positive reward relative to the distance
            # between the closest point detected by the affordances
            # and the end effector position
            distance = np.linalg.norm(tcp_pos - self.current_target)
            rew = rew + \
                (1 - (distance / self.banana_radio))
        return rew
