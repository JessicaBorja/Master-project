import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import sys
import os
import cv2
from omegaconf import OmegaConf
import pybullet as p
import math
from sklearn.cluster import DBSCAN
from matplotlib import cm

from sac_agent.sac import SAC
from sac_agent.sac_utils.utils import EpisodeStats, tt

from affordance_model.segmentator_centers import Segmentator
from affordance_model.datasets import get_transforms

from utils.cam_projections import pixel2world
from utils.img_utils import torch_to_numpy, overlay_mask, viz_aff_centers_preds


class Combined(SAC):
    def __init__(self, cfg, sac_cfg=None):
        super(Combined, self).__init__(**sac_cfg)
        self.affordance = cfg.target_search_aff
        self.aff_net_static_cam = self._init_static_cam_aff_net()
        self.writer = SummaryWriter(self.writer_name)
        self.cam_id = self._find_cam_id()
        self.transforms = get_transforms(
            cfg.transforms.validation,
            self.affordance.img_size)
        # initial angle
        self.target_orn = np.array([- math.pi, 0, - math.pi / 2])
        # self.env.get_obs()["robot_obs"][3:6]

        # Search for targets
        # Make target slightly(5cm) above actual target
        self._compute_target = self._env_compute_target
        # self._compute_target = self._aff_compute_target
        # self._compute_target = self._compute_target_centers
        self.area_center, self.target = self._compute_target()

        # Target specifics
        self.env.unwrapped.current_target = self.target
        self.eval_env.unwrapped.current_target = self.target
        self.radius = self.env.target_radius  # Distance in meters

    def _find_cam_id(self):
        for i, cam in enumerate(self.env.cameras):
            if "gripper" not in cam.name:
                return i
        return 0

    def _init_static_cam_aff_net(self):
        path = self.affordance.model_path
        aff_net = None
        if(os.path.exists(path)):
            hp = OmegaConf.to_container(self.affordance.hyperparameters)
            hp = OmegaConf.create(hp)
            aff_net = Segmentator.load_from_checkpoint(
                                path,
                                cfg=hp)
            aff_net.cuda()
            aff_net.eval()
            print("Static cam affordance model loaded (to find targets)")
        else:
            self.affordance = None
            path = os.path.abspath(path)
            print("Static cam aff. Path does not exist: %s" % path)
        return aff_net

    # Env real target pos
    def _env_compute_target(self):
        # This should come from static cam affordance later on
        target_pos, _ = self.env.get_target_pos()
        # 2 cm deviation
        target_pos = np.array(target_pos)
        target_pos += np.random.normal(loc=0, scale=0.002,
                                       size=(len(target_pos)))
        area_center = np.array(target_pos) \
            + np.array([0, 0, 0.07])

        # Visualize targets
        # p.removeAllUserDebugItems()
        # p.addUserDebugText("target_0",
        #                    textPosition=target_pos,
        #                    textColorRGB=[0, 1, 0])
        # p.addUserDebugText("a_center",
        #                    textPosition=area_center,
        #                    textColorRGB=[0, 1, 0])
        return area_center, target_pos

    # Clustering
    def _aff_compute_target(self):
        # Predictions for current observation
        cam = self.env.cameras[self.cam_id]
        obs = self.env.get_obs()
        img_obs = obs["rgb_obs"][self.cam_id]
        depth_obs = obs["depth_obs"][self.cam_id]
        cv2.imshow("img_obs", img_obs[:, :, ::-1])
        cv2.waitKey(1)
        orig_H, orig_W = img_obs.shape[:2]

        # Prediction from aff_model
        processed_obs = self.transforms(
                            torch.tensor(img_obs).permute(2, 0, 1))
        processed_obs = processed_obs.unsqueeze(1).cuda()
        aff_preds, probs, logits = self.aff_net_static_cam(processed_obs)
        aff_preds = aff_preds.cpu().detach().numpy()
        # (img_size, img_size, 1)
        aff_preds = np.transpose(aff_preds[0], (1, 2, 0))
        probs = probs[0].cpu().detach().numpy()
        probs = np.transpose(probs, (1, 2, 0))

        # Clustering
        # Predictions are between 0 and 1
        dbscan = DBSCAN(eps=3, min_samples=3)
        positives = np.argwhere(aff_preds > 0.3)
        cluster_outputs = []
        if(positives.shape[0] > 0):
            labels = dbscan.fit_predict(positives)
        else:
            return [0, 0, 0], [0, 0, 0]  # area_center, target_pos

        cluster_ids = np.unique(labels)

        # For printing the clusters
        colors = cm.jet(np.linspace(0, 1, len(cluster_ids)))
        colors = (colors * 255).astype("int")
        out_mask = np.zeros((orig_H, orig_W, 3))  # (64, 64, 3)

        # Find approximate center
        max_robustness = 0
        px_target = (0, 0)
        n_pixels = aff_preds.shape[0] * aff_preds.shape[1]
        out_img = img_obs[:, :, ::-1]
        for idx, c in enumerate(cluster_ids):
            cluster = positives[np.argwhere(labels == c).squeeze()]  # N, 3
            if(len(cluster.shape) == 1):
                cluster = np.expand_dims(cluster, 0)
            pixel_count = cluster.shape[0] / n_pixels

            # Visualize all cluster
            curr_mask = np.zeros((*aff_preds.shape[:2], 1))
            curr_mask[cluster[:, 0], cluster[:, 1]] = 255
            curr_mask = cv2.resize(curr_mask, (orig_H, orig_W))
            out_img = overlay_mask(curr_mask, out_img, tuple(colors[idx][:3]))

            # img size is 300 then we need int64 to cover all pixels
            scaled_cluster = np.argwhere(curr_mask > 0)
            mid_point = np.mean(scaled_cluster, 0).astype('int')
            u, v = mid_point[1], mid_point[0]
            # u = int(u * orig_W / aff_preds.shape[0])
            # v = int(v * orig_W / aff_preds.shape[0])
            # Unprojection
            world_pt = pixel2world(cam, u, v, depth_obs)

            # Softmax output from predicted affordances
            robustness = np.mean(probs[cluster[:, 0], cluster[:, 1]])

            if(robustness > max_robustness):
                target_pos = world_pt
                px_target = (u, v)
            c_out = {"px_center": (u, v),
                     "pixel_count": pixel_count,
                     "robustness": robustness}
            cluster_outputs.append(c_out)

        # Viz imgs
        for c in cluster_outputs:
            center = c["px_center"]
            out_img = cv2.drawMarker(out_img, center,
                                     (0, 0, 0),
                                     markerType=cv2.MARKER_CROSS,
                                     markerSize=5,
                                     line_type=cv2.LINE_AA)

        out_img = cv2.drawMarker(out_img, px_target,
                                 (0, 255, 0),
                                 markerType=cv2.MARKER_CROSS,
                                 markerSize=5,
                                 line_type=cv2.LINE_AA)

        depth_obs = cv2.drawMarker(depth_obs, px_target,
                                 (0, 255, 0),
                                 markerType=cv2.MARKER_CROSS,
                                 markerSize=5,
                                 line_type=cv2.LINE_AA)
        cv2.imshow("depth", depth_obs)
        cv2.imshow("clusters", out_img)
        cv2.waitKey(1)

        # 2 cm deviation
        target_pos = np.array(target_pos)
        target_pos += np.random.normal(loc=0, scale=0.002,
                                       size=(len(target_pos)))
        area_center = np.array(target_pos) \
            + np.array([0, 0, 0.05])
        return area_center, target_pos

    # Aff-center model
    def _compute_target_centers(self):
        # Get environment observation
        cam = self.env.cameras[self.cam_id]
        obs = self.env.get_obs()
        depth_obs = obs["depth_obs"][self.cam_id]
        orig_img = obs["rgb_obs"][self.cam_id]

        # Apply validation transforms
        img_obs = torch.tensor(orig_img).permute(2, 0, 1).unsqueeze(0).cuda()
        img_obs = self.transforms(img_obs)

        # Predict affordances and centers
        _, aff_probs, aff_mask, center_dir = self.forward(img_obs)
        aff_mask, center_dir, object_centers, object_masks = \
            self.predict(aff_mask, center_dir)

        # To numpy
        aff_probs = torch_to_numpy(aff_probs[0].permute(1, 2, 0))  # H, W, 2
        object_masks = torch_to_numpy(object_masks[0])  # H, W

        # Visualize predictions
        # viz_aff_centers_preds(orig_img, aff_mask, aff_probs, center_dir,
        #                       object_centers, object_masks)

        # Plot different objects
        if(len(object_centers) > 0):
            target_px = object_centers[0]
        else:
            return [0, 0, 0], [0, 0, 0]  # No center detected

        max_robustness = 0
        obj_class = np.unique(object_masks)[1:]
        obj_class = obj_class[obj_class != 0]  # remove background class

        # Look for most likely center
        for i, o in enumerate(object_centers):
            # Mean prob of being class 1 (foreground)
            robustness = np.mean(aff_probs[object_masks == obj_class[i], 1])
            if(robustness > max_robustness):
                max_robustness = robustness
                target_px = o

        # Convert back to observation size
        pred_shape = aff_probs.shape[:2]
        orig_shape = depth_obs.shape[:2]
        target_px = target_px.detach().cpu().numpy()
        target_px = (target_px * orig_shape / pred_shape).astype("int64")

        # world cord
        v, u = target_px
        # out_img = cv2.drawMarker(np.array(orig_img[:, :, ::-1]),
        #                          (u, v),
        #                          (0, 255, 0),
        #                          markerType=cv2.MARKER_CROSS,
        #                          markerSize=12,
        #                          line_type=cv2.LINE_AA)
        # cv2.imshow("out_img", out_img)

        # Compute depth
        target_pos = pixel2world(cam, u, v, depth_obs)

        # 2 cm deviation
        target_pos = np.array(target_pos)
        target_pos += np.random.normal(loc=0, scale=0.002,
                                       size=(len(target_pos)))
        area_center = np.array(target_pos) \
            + np.array([0, 0, 0.03])
        return area_center, target_pos

    def move_to_target(self, env, dict_obs, tcp_pos, a):
        target = a[0]
        # env.robot.apply_action(a)
        last_pos = target
        # When robot is moving and far from target
        while(np.linalg.norm(tcp_pos - target) > 0.01
              and np.linalg.norm(last_pos - tcp_pos) > 0.0005):
            last_pos = tcp_pos

            # Update position
            env.robot.apply_action(a)
            for i in range(8):
                env.p.stepSimulation()
                env.fps_controller.step()
            if(dict_obs):
                tcp_pos = env.get_obs()["robot_obs"][:3]
            else:
                tcp_pos = env.get_obs()[:3]

    def correct_position(self, env, s):
        dict_obs = False
        # Take current robot state
        if(isinstance(s, dict)):
            s = s["robot_obs"]
            dict_obs = True
        tcp_pos, gripper_action = s[:3], s[-1]

        # Compute target in case it moved
        # Area center is the target position + 5cm in z direction
        self.area_center, self.target = \
            self._compute_target()
        target = self.target

        # Set current_target in each episode
        env.unwrapped.current_target = target
        # self.eval_env.unwrapped.current_target = target

        # p.addUserDebugText("a_center",
        #                    textPosition=self.area_center,
        #                    textColorRGB=[0, 0, 1])
        if(np.linalg.norm(tcp_pos - target) > self.radius):
            up_target = [tcp_pos[0],
                         tcp_pos[1],
                         self.area_center[2] + 0.20]
            # Move up
            a = [up_target, self.target_orn, 1]
            self.move_to_target(env, dict_obs, tcp_pos, a)

            # Move to target
            tcp_pos = env.get_obs()["robot_obs"][:3]
            a = [self.area_center, self.target_orn, 1]
            self.move_to_target(env, dict_obs, tcp_pos, a)
            # p.addUserDebugText("target",
            #                    textPosition=target,
            #                    textColorRGB=[0, 0, 1])
        # as we moved robot, need to update target and obs
        # for rl policy
        return env, env.observation(env.get_obs())

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
            # Correct Position
            env, s = self.correct_position(env, s)
            while(episode_length < max_episode_length and not done):
                # sample action and scale it to action space
                a, _ = self._pi.act(tt(s), deterministic=True)
                a = a.cpu().detach().numpy()
                ns, r, done, info = env.step(a)
                if(render):
                    img = env.render()
                if(save_images):
                    # img, _ = self.find_target_center(env)
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
        episode = 1
        s = self.env.reset()
        episode_return, episode_length = 0, 0
        best_return, best_eval_return = -np.inf, -np.inf
        if(max_episode_length is None):
            max_episode_length = sys.maxsize  # "infinite"

        plot_data = {"actor_loss": [],
                     "critic_loss": [],
                     "ent_coef_loss": [], "ent_coef": []}

        _log_n_ep = log_interval//max_episode_length
        if(_log_n_ep < 1):
            _log_n_ep = 1

        # correct_every_ts = 20
        # Move to target position only one
        # Episode ends if outside of radius
        self.env, s = self.correct_position(self.env, s)
        for t in range(1, total_timesteps+1):
            s, done, episode_return, episode_length, plot_data, info = \
                self.training_step(s, t, episode_return, episode_length,
                                   plot_data)

            # End episode
            end_ep = done or (max_episode_length
                              and (episode_length >= max_episode_length))

            # Log interval (sac)
            if((t % log_interval == 0 and not self._log_by_episodes)
               or (self._log_by_episodes and end_ep
                   and episode % _log_n_ep == 0)):
                best_eval_return, plot_data = \
                     self._eval_and_log(self.writer, t, episode,
                                        plot_data, best_eval_return,
                                        n_eval_ep, max_episode_length)

            if(end_ep):
                best_return = \
                    self._on_train_ep_end(self.writer, t, episode,
                                          total_timesteps, best_return,
                                          episode_length, episode_return)
                # Reset everything
                episode += 1
                episode_return, episode_length = 0, 0
                s = self.env.reset()
                self.env, s = self.correct_position(self.env, s)
