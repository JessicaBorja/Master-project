import sys
import hydra
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from utils.utils import register_env
from sac_agent.sac import SAC
from affordance_model.segmentator import Segmentator
from affordance_model.datasets import get_transforms
from affordance_model.utils.utils import visualize
import os
import gym
import cv2
import torch
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import pybullet as p
import math
register_env()


class Combined(SAC):
    def __init__(self, cfg, sac_cfg=None):
        super(Combined, self).__init__(**sac_cfg)
        self.affordance = cfg.affordance
        self.writer = SummaryWriter(self.writer_name)
        self.area_center = [0, 0, 0]
        self.radius = 10
        self.aff_net = self._init_aff_net()
        self.cam_id = self._find_cam_id()
        self.transforms = get_transforms(cfg.transforms.validation)

    def _find_cam_id(self):
        for i, cam in enumerate(self.env.cameras):
            if "gripper" not in cam.name:
                return i
        return 0

    def _init_aff_net(self):
        path = self.affordance.static_cam.model_path
        if(os.path.exists(path)):
            aff_net = Segmentator.load_from_checkpoint(
                                path,
                                cfg=self.affordance.hyperparameters)
            aff_net.cuda()
            aff_net.eval()
            print("Static cam affordance model loaded")
        else:
            self.affordance = None
            print("Path does not exist: %s" % path)
        return aff_net

    def find_target_center(self):
        rgb, depth = self.env.get_camera_obs()
        # Values from static camera
        static_img = rgb[self.cam_id][:, :, ::-1].copy() # H, W, C
        static_depth = depth[self.cam_id]
        x = torch.from_numpy(static_img).permute(2, 0, 1).unsqueeze(0)
        x = self.transforms(x).cuda()
        mask = self.aff_net(x)
        # Visualization
        out_img = visualize(mask, static_img, False)

        # Reshape to fit large img_size
        orig_H, orig_W = static_img.shape[:2]
        mask_np = mask.detach().cpu().numpy()
        if(mask_np.shape[1] > 1):
            # mask_np in set [0,1]
            mask_np = np.argmax(mask_np, axis=1)
        mask_np = mask_np.astype('uint8').squeeze()
        mask_np = cv2.resize(mask_np,
                             dsize=(orig_H, orig_W),
                             interpolation=cv2.INTER_CUBIC)

        # Make clusters
        dbscan = DBSCAN(eps=3, min_samples=3)
        positives = np.argwhere(mask_np > 0.5)
        labels = dbscan.fit_predict(positives)

        fig, ax = plt.subplots()
        ax.imshow(static_img[:, :, ::-1])
        scatter = ax.scatter(positives[:, 1], positives[:, 0],
                             c=labels, cmap="hsv")

        # Find approximate center
        points_3d = []
        for idx, c in enumerate(np.unique(labels)):
            mid_point = np.mean(positives[np.argwhere(labels == c)], 0)[0]
            ax.scatter(mid_point[1], mid_point[0], s=25, marker='x', c='k')

            mid_point = mid_point.astype('uint8')
            pixel_pt = (mid_point[1], mid_point[0])
            out_img = cv2.drawMarker(out_img, pixel_pt,
                                     (0, 0, 0),
                                     markerType=cv2.MARKER_CROSS,
                                     markerSize=5,
                                     line_type=cv2.LINE_AA)
            depth = static_depth[pixel_pt[0], pixel_pt[1]]
            world_pt = self.pixel2world(pixel_pt[0], pixel_pt[1], depth, str(idx))
            points_3d.append(world_pt)

        ax.axis('off')
        # plt.show()

        # Viz imgs
        cv2.imshow("clusters", out_img)
        cv2.imshow("depth", static_depth)
        cv2.waitKey(1)

    def getRayFromTo(mouseX, mouseY):
        width, height, viewMat, projMat, cameraUp, camForward, horizon, vertical, _, _, dist, camTarget = p.getDebugVisualizerCamera()
        camPos = [
            camTarget[0] - dist * camForward[0], camTarget[1] - dist * camForward[1],
            camTarget[2] - dist * camForward[2]
        ]
        farPlane = 10000
        rayForward = [(camTarget[0] - camPos[0]), (camTarget[1] - camPos[1]), (camTarget[2] - camPos[2])]
        lenFwd = math.sqrt(rayForward[0] * rayForward[0] + rayForward[1] * rayForward[1] +
                            rayForward[2] * rayForward[2])
        invLen = farPlane * 1. / lenFwd
        rayForward = [invLen * rayForward[0], invLen * rayForward[1], invLen * rayForward[2]]
        rayFrom = camPos
        oneOverWidth = float(1) / float(width)
        oneOverHeight = float(1) / float(height)

        dHor = [horizon[0] * oneOverWidth, horizon[1] * oneOverWidth, horizon[2] * oneOverWidth]
        dVer = [vertical[0] * oneOverHeight, vertical[1] * oneOverHeight, vertical[2] * oneOverHeight]
        ortho = [
            -0.5 * horizon[0] + 0.5 * vertical[0] + float(mouseX) * dHor[0] - float(mouseY) * dVer[0],
            -0.5 * horizon[1] + 0.5 * vertical[1] + float(mouseX) * dHor[1] - float(mouseY) * dVer[1],
            -0.5 * horizon[2] + 0.5 * vertical[2] + float(mouseX) * dHor[2] - float(mouseY) * dVer[2]
        ]

        rayTo = [
            rayFrom[0] + rayForward[0] + ortho[0], rayFrom[1] + rayForward[1] + ortho[1],
            rayFrom[2] + rayForward[2] + ortho[2]
        ]
        lenOrtho = math.sqrt(ortho[0] * ortho[0] + ortho[1] * ortho[1] + ortho[2] * ortho[2])
        alpha = math.atan(lenOrtho / farPlane)
        return rayFrom, rayTo, alpha

    def pixel2world(self, u, v, depth, debug_str):
        cam = self.env.cameras[self.cam_id]
        persp_m = np.array(cam.projectionMatrix).reshape((4, 4)).T
        view_m = np.array(cam.viewMatrix).reshape((4, 4)).T

        # Perspective proj matrix
        d = depth * cam.farval
        u, v = depth*(u / cam.width), depth*(1 - (v / cam.height))
        px_pt = np.array([u, v, d, 1])
        px_pt[:3] = (px_pt[:3] * 2) - 1
        pix_world_tran = np.linalg.inv(persp_m @ view_m) @ px_pt
        pix_world_tran = pix_world_tran / pix_world_tran[-1]
        # world_pix_tran = persp_m @ view_m @ point
        # world_pix_tran = world_pix_tran / world_pix_tran[-1]  # divide by w
        # world_pix_tran[:3] = (world_pix_tran[:3] + 1)/2
        world_pos = pix_world_tran[:3]

        p.addUserDebugText(debug_str,
                           textPosition=world_pos,
                           textColorRGB=[1, 0, 0])
        # x, y = world_pix_tran[0]*cam.width, (1-world_pix_tran[1])*cam.height
        return world_pos

    def correct_position(self, s):
        if(isinstance(s, dict)):
            tcp_pos = s["robot_pos"][:3]
        else:
            tcp_pos = s[:3]
        # If outside the area
        if(np.linalg.norm(tcp_pos - self.area_center) > self.radius):
            self.env.robot.apply_action(self.area_center)

    def learn(self, total_timesteps=10000, log_interval=100,
              max_episode_length=None, n_eval_ep=5):
        if not isinstance(total_timesteps, int):   # auto
            total_timesteps = int(total_timesteps)
        episode = 0
        s = self.env.reset()
        episode_return, episode_length = 0, 0
        best_return, best_eval_return = -np.inf, -np.inf
        if(max_episode_length is None):
            max_episode_length = sys.maxsize  # "infinite"

        plot_data = {"actor_loss": [],
                     "critic_loss": [],
                     "ent_coef_loss": [], "ent_coef": []}
        for t in range(1, total_timesteps+1):
            self.correct_position(s)
            s, done, episode_return, episode_length, plot_data, info = \
                self.training_step(s, t, episode_return, episode_length,
                                   plot_data)

            # End episode
            if(done or (max_episode_length and
                        (episode_length >= max_episode_length))):
                best_return = \
                    self._on_train_ep_end(self.writer, t, episode,
                                          total_timesteps, best_return,
                                          episode_length, episode_return)
                # Reset everything
                episode += 1
                episode_return, episode_length = 0, 0
                s = self.env.reset()

            # Log interval (sac)
            if(t % log_interval == 0):
                best_eval_return, plot_data = \
                     self._log_eval_stats(self.writer, t, episode,
                                          plot_data, best_eval_return,
                                          n_eval_ep, max_episode_length)


@hydra.main(config_path="./config", config_name="cfg_combined")
def main(cfg):

    env = gym.make("VREnv-v0", **cfg.env).env
    model_name = cfg.model_name
    sac_cfg = {"env": env,
               "model_name": model_name,
               "save_dir": cfg.agent.save_dir,
               "net_cfg": cfg.agent.net_cfg,
               **cfg.agent.hyperparameters}

    model = Combined(cfg, sac_cfg=sac_cfg)
    for i in range(10000):
        action = env.action_space.sample()
        ns, r, d, info = env.step(action)
        model.find_target_center()

    env.close()


if __name__ == "__main__":
    main()
