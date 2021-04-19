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
from utils.env_processing_wrapper import EnvWrapper
from sac_agent.sac_utils.utils import EpisodeStats, tt
import glm
register_env()


class Combined(SAC):
    def __init__(self, cfg, sac_cfg=None):
        super(Combined, self).__init__(**sac_cfg)
        self.affordance = cfg.affordance
        self.writer = SummaryWriter(self.writer_name)
        self.target_orn = \
            self.env.get_obs()["robot_obs"][3:6]

        # Make target slightly(5cm) above actual target
        self.area_center = self.compute_target()
        self.radius = 0.10  # Distance in meters
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
        aff_net = None
        if(os.path.exists(path)):
            aff_net = Segmentator.load_from_checkpoint(
                                path,
                                cfg=self.affordance.hyperparameters)
            aff_net.cuda()
            aff_net.eval()
            print("Static cam affordance model loaded (to find targets)")
        else:
            self.affordance = None
            print("Path does not exist: %s" % path)
        return aff_net

    def find_target_center(self):
        rgb, depth = self.env.get_camera_obs()
        # Values from static camera
        static_img = rgb[self.cam_id][:, :, ::-1].copy()  # H, W, C
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

    def world2pixel(self, point, cam):
        # https://github.com/bulletphysics/bullet3/issues/1952
        # reshape to get homogeneus transform
        persp_m = np.array(cam.projectionMatrix).reshape((4, 4)).T
        view_m = np.array(cam.viewMatrix).reshape((4, 4)).T

        # Perspective proj matrix
        world_pix_tran = persp_m @ view_m @ point
        world_pix_tran = world_pix_tran / world_pix_tran[-1]  # divide by w
        world_pix_tran[:3] = (world_pix_tran[:3] + 1)/2
        x, y = world_pix_tran[0]*cam.width, (1-world_pix_tran[1])*cam.height
        x, y = np.floor(x).astype(int), np.floor(y).astype(int)
        return (x, y)

    def pixel2world(self, u, v, depth, debug_str, cam):
        persp_m = np.array(cam.projectionMatrix).reshape((4, 4), order='F')
        view_m = np.array(cam.viewMatrix).reshape((4, 4), order='F')

        # Perspective proj matrix
        # d = depth * cam.farval
        # u, v = u / cam.width, 1 - (v / cam.height)
        # px_pt = np.array([u, v, depth * cam.farval, 1])
        # px_pt[:3] = (px_pt[:3] * 2) - 1
        # pix_world_tran = np.linalg.inv(persp_m @ view_m) @ px_pt
        # pix_world_tran = pix_world_tran / pix_world_tran[-1]
        # world_pos = pix_world_tran[:3]

        far = cam.farval
        win = glm.vec3(float(u), float(v), depth*far)
        viewport = glm.vec4(0, 0, cam.width, cam.height)
        view_m, persp_m = glm.mat4(view_m), glm.mat4(persp_m)
        world_pos = glm.unProject(win, view_m, persp_m, viewport)
        world_pos = np.array([world_pos[0], world_pos[1], world_pos[2]])
        p.addUserDebugText(debug_str,
                            textPosition=world_pos,
                            textColorRGB=[1, 0, 0])
        return world_pos

    def compute_target(self):
        target_pos, _ = self.env.get_target_pos()
        area_center = np.array(target_pos) \
            + np.array([0, 0, 0.05])
        return area_center

    def correct_position(self, env, s):
        dict_obs = False
        # Take current robot state
        if(isinstance(s, dict)):
            s = s["robot_obs"]
            dict_obs = True
        tcp_pos, gripper = s[:3], s[-1]

        # Compute target in case it moved
        self.area_center = self.compute_target()
        _envts = 0  # to set a timeout
        up = False
        # If outside the area
        while(np.linalg.norm(tcp_pos - self.area_center) > self.radius
              and _envts < 120):
            # Update position
            if(dict_obs):
                tcp_pos = env.get_obs()["robot_obs"][:3]
            else:
                tcp_pos = env.get_obs()[:3]

            # First move up
            if(np.abs(tcp_pos[2] - (self.area_center[2] + 0.1)) > 0.1
               and not up):
                target_pos = [tcp_pos[0],
                              tcp_pos[1],
                              self.area_center[2] + 0.2]
                a = [target_pos, self.target_orn, gripper]
            else:
                up = True

            # Once we've gone up, always move to target
            if(up):
                a = [self.area_center, self.target_orn, gripper]

            # Apply corrective action
            env.robot.apply_action(a)
            env.p.stepSimulation()
            env.fps_controller.step()
            _envts += 1

    def evaluate(self, env, max_episode_length=150, n_episodes=5,
                 print_all_episodes=False, render=False, save_images=False):
        stats = EpisodeStats(episode_lengths=[],
                             episode_rewards=[],
                             validation_reward=[])
        im_lst = []
        for episode in range(n_episodes):
            s = env.reset()
            episode_length, episode_return = 0, 0
            done = False
            while(episode_length < max_episode_length and not done):
                # Correct Position
                self.correct_position(env, s)
                # sample action and scale it to action space
                a, _ = self._pi.act(tt(s), deterministic=True)
                a = a.cpu().detach().numpy()
                ns, r, done, info = env.step(a)
                if(render):
                    img = env.render()
                if(save_images):
                    img = env.render('rgb_array')
                    im_lst.append(img)
                s = ns
                episode_return += r
                episode_length += 1
            stats.episode_rewards.append(episode_return)
            stats.episode_lengths.append(episode_length)
            if(print_all_episodes):
                print("Episode %d, Return: %.3f" % (episode, episode_return))

        # Save images
        if(save_images):
            import cv2
            os.makedirs("./frames/")
            for idx, im in enumerate(im_lst):
                cv2.imwrite("./frames/image_%04d.jpg" % idx, im)
        # mean and print
        mean_reward = np.mean(stats.episode_rewards)
        reward_std = np.std(stats.episode_rewards)
        mean_length = np.mean(stats.episode_lengths)
        length_std = np.std(stats.episode_lengths)

        self.log.info(
            "Mean return: %.3f +/- %.3f, " % (mean_reward, reward_std) +
            "Mean length: %.3f +/- %.3f, over %d episodes" %
            (mean_length, length_std, n_episodes))
        return mean_reward, mean_length

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
        # correct_every_ts = 20
        for t in range(1, total_timesteps+1):
            self.correct_position(self.env, s)
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
                     self._eval_and_log(self.writer, t, episode,
                                        plot_data, best_eval_return,
                                        n_eval_ep, max_episode_length)


@hydra.main(config_path="./config", config_name="cfg_combined")
def main(cfg):

    training_env = gym.make("VREnv-v0", **cfg.env).env
    eval_env = gym.make("VREnv-v0", **cfg.eval_env).env
    training_env = EnvWrapper(training_env, train=True,
                              affordance=cfg.affordance,
                              **cfg.env_wrapper)
    eval_env = EnvWrapper(eval_env,
                          affordance=cfg.affordance,
                          **cfg.env_wrapper)
    sac_cfg = {"env": training_env,
               "eval_env": eval_env,
               "model_name": cfg.model_name,
               "save_dir": cfg.agent.save_dir,
               "net_cfg": cfg.agent.net_cfg,
               **cfg.agent.hyperparameters}

    model = Combined(cfg, sac_cfg=sac_cfg)
    model.learn(**cfg.agent.learn_config)
    training_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
