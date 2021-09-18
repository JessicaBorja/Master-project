import collections
import time

import numpy as np
from torch.utils.tensorboard import SummaryWriter
import sys
import os
import cv2
import math
import math
from sac_agent.sac import SAC
from sac_agent.sac_utils.utils import tt

from affordance_model.datasets import get_transforms
from combined.target_search import TargetSearch

class FpsController:
    def __init__(self, freq):
        self.loop_time = 1.0 / freq
        self.prev_time = time.time()

    def step(self):
        current_time = time.time()
        delta_t = current_time - self.prev_time
        if delta_t < self.loop_time:
            time.sleep(self.loop_time - delta_t)
        self.prev_time = time.time()

class Combined(SAC):
    def __init__(self, cfg, sac_cfg=None,
                 rand_target=False, target_search_mode="env"):
        super(Combined, self).__init__(**sac_cfg)
        self.writer = SummaryWriter(self.writer_name)
        _aff_transforms = get_transforms(
            cfg.affordance.transforms.validation,
            cfg.target_search_aff.img_size)

        # initial angle
        _initial_obs = self.env.reset()["robot_obs"]
        self.origin = _initial_obs[:3]
        self.target_orn = np.array([math.pi, 0, 0])

        # Box pos world frame
        # self.box_pos = self.env.objects["bin"]["initial_pos"] # np.array([0, 0, 0])

        # To enumerate static cam preds on target search
        self.global_obs_it = 0
        self.no_detected_target = 0

        args = {"initial_pos": self.origin,
                "aff_cfg": cfg.target_search_aff,
                "aff_transforms": _aff_transforms,
                "rand_target": rand_target}
        self.target_search = TargetSearch(self.env,
                                          main_cfg=cfg,
                                          mode="real_world",
                                          **args)

        self.target_pos, _ = self.target_search.compute()

        # Target specifics
        self.env.target_search = self.target_search
        self.env.current_target = self.target_pos
        self.env.unwrapped.current_target = self.target_pos
        self.eval_env = self.env
        self.above_obj = 0.06

    def move_to_box(self, env, sample=False):
        # Box does not move
        r_obs = env.get_obs()["robot_obs"]
        tcp_pos, _ = r_obs[:3], r_obs[3:6]
        center_x, center_y = self.box_pos[:2]
        if(sample):
            x_pos = np.random.uniform(center_x + 0.06, center_x - 0.06)
            y_pos = np.random.uniform(center_y - 0.06, center_y + 0.06)
        else:
            x_pos = center_x
            y_pos = center_y
        box_pos = [x_pos, y_pos, 0.65]

        env.reset(box_pos, self.target_orn)

    def detect_and_correct(self, env, p_dist=None):
        self.env.reset()
        # Compute target in case it moved
        # Area center is the target position + 5cm in z direction
        target_pos, no_target = \
            self.target_search.compute(env,
                                       self.global_obs_it,
                                       p_dist=p_dist)
        if(no_target):
            self.no_detected_target += 1
            input("No object detected. Please rearrange table.")
            return self.detect_and_correct(env, p_dist)

        robot_target_pos = target_pos.copy()
        robot_target_pos[2] += self.above_obj
        target_orn = self.target_orn.copy()
        target_orn[2] += np.random.uniform(-1, 1) * np.radians(30)
        obs = self.env.reset(robot_target_pos, target_orn)
        self.env.curr_detected_obj = target_pos
        return env, obs, no_target

    # RL Policy
    def learn(self, total_timesteps=10000, log_interval=100,
              max_episode_length=None, n_eval_ep=5):
        if not isinstance(total_timesteps, int):   # auto
            total_timesteps = int(total_timesteps)
        episode_return, episode_length = 0, 0
        if(max_episode_length is None):
            max_episode_length = sys.maxsize  # "infinite"

        plot_data = {"actor_loss": [],
                     "critic_loss": [],
                     "ent_coef_loss": [], "ent_coef": []}

        _log_n_ep = log_interval//max_episode_length
        if(_log_n_ep < 1):
            _log_n_ep = 1

        # Move to target position only one
        # Episode ends if outside of radius
        self.env, s, no_target = self.detect_and_correct(self.env)

        #fps = FpsController(20)

        for t in range(1, total_timesteps+1):
            t = time.time()
            s, done, success, episode_return, episode_length, plot_data, info = \
                self.training_step(s, self.curr_ts, episode_return, episode_length,
                                   plot_data)

            # End episode
            timeout = (max_episode_length
                       and (episode_length >= max_episode_length))
            end_ep = timeout or done

            # Log interval (sac)
            if((t % log_interval == 0 and not self._log_by_episodes)
               or (self._log_by_episodes and end_ep
                   and self.episode % _log_n_ep == 0)):
                self.best_eval_return, self.most_tasks, plot_data = \
                    self._eval_and_log(self.writer, self.curr_ts, self.episode,
                                       plot_data, self.most_tasks,
                                       self.best_eval_return,
                                       n_eval_ep, max_episode_length)

            if(end_ep):
                self.best_return = \
                    self._on_train_ep_end(self.writer, self.curr_ts, self.episode,
                                          total_timesteps, self.best_return,
                                          episode_length, episode_return,
                                          success)
                self.last_n_train_success.append(success)
                # Reset everything
                self.episode += 1
                self.env.gripper_cam_imgs={}
                self.env.obs_it=0
                episode_return, episode_length = 0, 0
                # Go to origin then look for obj
                self.env, s, no_target = self.detect_and_correct(self.env)

            # fps.step()
            # print(1 / (time.time() - t))
            self.curr_ts += 1

    def evaluate(self, env, max_episode_length=150, n_episodes=5,
                 print_all_episodes=False, save_images=False, deterministic=True):
        self.log.info("STARTING EVALUATION")
        ep_returns, ep_lengths = [], []
        no_target = True
        while(no_target):
            s = env.reset()
            target_pos, no_target, center_targets = \
                self.target_search.compute(env,
                                           self.global_obs_it,
                                           return_all_centers=True)
            if(no_target):
                input("No object detected. Please rearrange table.")

        run_n_episodes = len(center_targets)
        ep_success = []
        # One episode per task
        for episode in range(run_n_episodes):
            env.reset()
            target_pos = center_targets[episode]
            env.curr_detected_obj = target_pos
            episode_length, episode_return = 0, 0
            done = False
            target_pos[-1] += self.above_obj
            s = env.reset(target_pos, self.target_orn)
            while(episode_length < max_episode_length and not done):
                # sample action and scale it to action space
                a, _ = self._pi.act(tt(s), deterministic=deterministic)
                a = a.cpu().detach().numpy()
                ns, r, done, info = env.step(a)
                s = ns
                episode_return += r
                episode_length += 1
            success = info['success']
            ep_success.append(success)
            ep_returns.append(episode_return)
            ep_lengths.append(episode_length)

        # mean and print
        mean_reward, reward_std = np.mean(ep_returns), np.std(ep_returns)
        mean_length, length_std = np.mean(ep_lengths), np.std(ep_lengths)

        self.log.info(
            "Success: %d/%d " % (np.sum(ep_success), len(ep_success)) +
            "Mean return: %.3f +/- %.3f, " % (mean_reward, reward_std) +
            "Mean length: %.3f +/- %.3f, over %d episodes" %
            (mean_length, length_std, n_episodes))
        self.log.info("EVALUATION END")
        return mean_reward, mean_length, ep_success

    # Only applies to tabletop
    def tidy_up(self, env, max_episode_length=100):
        tasks = []
        # get from static cam affordance
        if(env.task == "pickup"):
            tasks = list(self.env.objects.keys())
            tasks.remove("table")
            tasks.remove("bin")
            n_tasks = len(tasks)

        ep_success = []
        total_ts = 0
        s = env.reset()
        self.no_detected_target = 0
        # Set total timeout to timeout per task times all tasks + 1
        while(total_ts <= max_episode_length // 2 * n_tasks
              and self.no_detected_target < 3):
            episode_length, episode_return = 0, 0
            done = False
            # Search affordances and correct position:
            env, s, no_target = self.detect_and_correct(env, self.env.get_obs())
            if(no_target):
                # If no target model will move to initial position.
                # Search affordance from this position again
                self.detect_and_correct(env, self.env.get_obs())

            # If it did not find a target again, terminate everything
            while(episode_length < max_episode_length // 2
                  and self.no_detected_target < 3
                  and not done):
                # sample action and scale it to action space
                a, _ = self._pi.act(tt(s), deterministic=True)
                a = a.cpu().detach().numpy()
                ns, r, _, info = env.step(a)
                done = r >= 200
                s = ns
                episode_return += r
                episode_length += 1
                total_ts += 1
            if(episode_return >= 200 and env.task == "pickup"):
                success = self.eval_grasp_success(env)
            else:
                success = False
            ep_success.append(success)
        self.save_images(env)
        self.log.info("Success: %d/%d " % (np.sum(ep_success), len(ep_success)))
        return ep_success

    def eval_grasp_success(self, env, any=False):
        self.move_to_box(env)
        if(any):
            success = False
            for name in env.table_objs:
                if(env.obj_in_box(env.objects[name])):
                    success = True
        else:
            success = env.obj_in_box(env.objects[env.target])
        return success

    # Save images
    def save_images(self, env):
        # Write all images
        if(env.save_images):
            # for idx, im in enumerate(self.im_lst):
            #     cv2.imwrite("./frames/image_%04d.jpg" % idx, im)
            for name, im in self.target_search.static_cam_imgs.items():
                head, _ = os.path.split(name)
                if(not os.path.exists(head)):
                    os.makedirs(head)
                cv2.imwrite(name, im)
            for name, im in env.gripper_cam_imgs.items():
                head, _ = os.path.split(name)
                if(not os.path.exists(head)):
                    os.makedirs(head)
                cv2.imwrite(name, im)
