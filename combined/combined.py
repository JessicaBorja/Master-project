import sys
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sac_agent.sac import SAC
from affordance_model.segmentator import Segmentator
from affordance_model.datasets import get_transforms
import os
import cv2
from sac_agent.sac_utils.utils import EpisodeStats, tt
from omegaconf import OmegaConf


class Combined(SAC):
    def __init__(self, cfg, sac_cfg=None):
        super(Combined, self).__init__(**sac_cfg)
        self.affordance = cfg.affordance
        self.writer = SummaryWriter(self.writer_name)
        self.target_orn = \
            self.env.get_obs()["robot_obs"][3:6]

        # Make target slightly(5cm) above actual target
        self.area_center, self.target = self.compute_target()
        self.env.current_target = self.target
        self.eval_env.current_target = self.target
        self.radius = self.env.banana_radio  # Distance in meters
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
            hp = OmegaConf.to_container(self.affordance.hyperparameters)
            hp['unet_cfg'].pop('decoder_channels')
            hp['unet_cfg']['decoder_channels'] = [256, 128, 64, 32]
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
            print("Path does not exist: %s" % path)
        return aff_net

    def compute_target(self):
        # This should come from static cam affordance later on
        target_pos, _ = self.env.get_target_pos()
        # 2 cm deviation
        target_pos = np.array(target_pos)
        target_pos += np.random.normal(loc=0, scale=0.02,
                                       size=(len(target_pos)))
        area_center = np.array(target_pos) \
            + np.array([0, 0, 0.05])
        return area_center, target_pos

    def move_to_target(self, env, dict_obs, tcp_pos, a):
        target = a[0]
        env.robot.apply_action(a)
        last_pos = target
        # When robot is moving and far from target
        while(np.linalg.norm(tcp_pos - target) > 0.02
              and np.linalg.norm(last_pos - tcp_pos) > 0.005):
            last_pos = tcp_pos

            # Update position
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
        tcp_pos, gripper = s[:3], s[-1]

        # Compute target in case it moved
        # Area center is the target position + 5cm in z direction
        # self.area_center, _ = self.compute_target()
        target = self.target
        if(np.linalg.norm(tcp_pos - target) > self.radius):
            up_target = [tcp_pos[0],
                         tcp_pos[1],
                         self.area_center[2] + 0.10]
            # Move up
            a = [up_target, self.target_orn, gripper]
            self.move_to_target(env, dict_obs, tcp_pos, a)

            # Move to target
            a = [self.area_center, self.target_orn, gripper]
            self.move_to_target(env, dict_obs, tcp_pos, a)

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
            self.correct_position(env, s)
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
        # Move to target position only one
        # Episode ends if outside of radius
        self.correct_position(self.env, s)
        for t in range(1, total_timesteps+1):
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
                self.correct_position(self.env, s)

            # Log interval (sac)
            if(t % log_interval == 0):
                best_eval_return, plot_data = \
                     self._eval_and_log(self.writer, t, episode,
                                        plot_data, best_eval_return,
                                        n_eval_ep, max_episode_length)
