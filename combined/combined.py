import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import sys
import os
import cv2
import pybullet as p
import math
from sac_agent.sac import SAC
from sac_agent.sac_utils.utils import EpisodeStats, tt


from affordance_model.datasets import get_transforms
from combined.target_search import TargetSearch


class Combined(SAC):
    def __init__(self, cfg, sac_cfg=None,
                 rand_target=False, target_search_mode="env"):
        super(Combined, self).__init__(**sac_cfg)
        self.writer = SummaryWriter(self.writer_name)
        _cam_id = self._find_cam_id()
        _aff_transforms = get_transforms(
            cfg.affordance.transforms.validation,
            cfg.target_search_aff.img_size)
        # initial angle
        _initial_obs = self.env.get_obs()["robot_obs"]
        _initial_pos = _initial_obs[:3]
        _initial_orn = _initial_obs[3:6]
        if(self.env.task == "pickup"):
            self.target_orn = np.array([- math.pi, 0, - math.pi / 2])
        else:
            self.target_orn = _initial_orn

        # to save images
        self.im_lst = []

        # To enumerate static cam preds on target search
        self.global_obs_it = 0
        self.no_detected_target = 0

        args = {"cam_id": _cam_id,
                "initial_pos": _initial_pos,
                "aff_cfg": cfg.target_search_aff,
                "aff_transforms": _aff_transforms,
                "rand_target": rand_target}
        _class_label = self.get_task_label()
        self.target_search = TargetSearch(self.env,
                                          class_label=_class_label,
                                          mode=target_search_mode,
                                          **args)
        if(self.env.task == "pickup"):
            self.box_mask, self.box_3D_end_points = \
                self.target_search.get_box_pos_mask(self.env)

        self.target_pos, _ = self.target_search.compute()

        # Target specifics
        self.env.unwrapped.current_target = self.target_pos
        self.eval_env.unwrapped.current_target = self.target_pos
        self.radius = self.env.target_radius  # Distance in meters

    def get_task_label(self):
        task = self.env.task
        if(task == "hinge"):
            return 1
        elif(task == "drawer"):
            return 2
        elif(task == "slide"):
            return 3
        else:  # pickup
            return None

    def _find_cam_id(self):
        for i, cam in enumerate(self.env.cameras):
            if "static" in cam.name:
                return i
        return 0

    # Model based methods
    def move_to_target(self, env, tcp_pos, a, dict_obs=True):
        target = a[0]
        # env.robot.apply_action(a)
        last_pos = target
        # When robot is moving and far from target
        while(np.linalg.norm(tcp_pos - target) > 0.01
              and np.linalg.norm(last_pos - tcp_pos) > 0.0005):
            last_pos = tcp_pos

            # Update position
            for i in range(8):
                env.robot.apply_action(a)
                env.p.stepSimulation(physicsClientId=env.cid)
            if(dict_obs):
                tcp_pos = env.get_obs()["robot_obs"][:3]
            else:
                tcp_pos = env.get_obs()[:3]

            # if(self.env.save_images):
            #     im = env.render()
            #     # self.im_lst.append(im)
            #     cv2.imwrite("./frames/image_%04d.jpg" % self.global_obs_it, im)
            #     self.global_obs_it += 1
        return tcp_pos  # end pos

    def move_to_box(self, env):
        # Box does not move
        r_obs = env.get_obs()["robot_obs"]
        tcp_pos, _ = r_obs[:3], r_obs[3:6]
        top_left, bott_right = self.box_3D_end_points
        x_pos = np.random.uniform(top_left[0] + 0.07, bott_right[0] - 0.07)
        y_pos = np.random.uniform(top_left[1] - 0.07, bott_right[1] + 0.07)
        box_pos = [x_pos, y_pos, 0.65]

        # Move up
        up_target = [*tcp_pos[:2], box_pos[2] + 0.2]
        a = [up_target, self.target_orn, -1]  # -1 means closed
        tcp_pos = self.move_to_target(env, tcp_pos, a, dict_obs=True)

        # Move to obj up
        up_target = [*box_pos[:2], tcp_pos[-1]]
        a = [up_target, self.target_orn, -1]  # -1 means closed
        tcp_pos = self.move_to_target(env, tcp_pos, a, dict_obs=True)

        # Move down
        box_pos = [*box_pos[:2], tcp_pos[-1] - 0.12]
        a = [box_pos, self.target_orn, -1]  # -1 means closed
        tcp_pos = env.get_obs()["robot_obs"][:3]
        self.move_to_target(env, tcp_pos, a, dict_obs=True)

        # Get new position and orientation
        # pos, z angle, action = open gripper
        tcp_pos = env.get_obs()["robot_obs"][:3]
        a = [tcp_pos, self.target_orn, 1]  # drop object
        for i in range(8):  # 8 steps
            env.robot.apply_action(a)
            for i in range(8):  # 1 rl steps
                env.p.stepSimulation()
                env.fps_controller.step()
            # if(self.env.save_images):
            #     im = env.render()
            #     cv2.imwrite("./frames/image_%04d.jpg" % self.global_obs_it,
            #                 im)
            #     self.global_obs_it += 1

    def correct_position(self, env, s):
        dict_obs = False
        # Take current robot state
        if(isinstance(s, dict)):
            s = s["robot_obs"]
            dict_obs = True
        tcp_pos, gripper_action = s[:3], s[-1]

        # Compute target in case it moved
        # Area center is the target position + 5cm in z direction
        self.target_pos, no_target = \
            self.target_search.compute(env, self.global_obs_it)
        if(no_target):
            self.no_detected_target += 1
        target_pos = self.target_pos

        # Set current_target in each episode
        env.unwrapped.current_target = target_pos
        # self.eval_env.unwrapped.current_target = target

        # p.addUserDebugText("a_center",
        #                    textPosition=self.target_pos,
        #                    textColorRGB=[0, 0, 1])
        if(np.linalg.norm(tcp_pos - target_pos) > self.radius):
            if(env.task == "pickup" or env.task == "drawer"):
                # To never collide with the box
                z_value = max(self.target_pos[2] + 0.08, 0.76)
                up_target = [tcp_pos[0],
                             tcp_pos[1],
                             z_value]
                # Move up from starting pose
                a = [up_target, self.target_orn, 1]
                tcp_pos = self.move_to_target(env, tcp_pos, a, dict_obs)

                # Move to target
                reach_target = [*self.target_pos[:2], tcp_pos[-1]]
                a = [reach_target, self.target_orn, 1]
                tcp_pos = self.move_to_target(env, tcp_pos, a, dict_obs)
                if(self.target_search.mode == "env"):
                    # Environment returns the center of mass..
                    move_to = self.target_pos + np.array([0, 0, 0.07])
                else:
                    # Affordances detect the surface of an object
                    move_to = self.target_pos + np.array([0, 0, 0.05])
            else:
                # Move in x-z dir
                # x_target = [self.target_pos[0], tcp_pos[1], self.target_pos[2]]
                # a = [x_target, self.target_orn, 1]
                # tcp_pos = self.move_to_target(env, tcp_pos, a, dict_obs)

                # Move up y dir
                move_to = self.target_pos

            # Move to target
            a = [move_to, self.target_orn, 1]
            self.move_to_target(env, tcp_pos, a, dict_obs)
            # p.addUserDebugText("target",
            #                    textPosition=target,
            #                    textColorRGB=[0, 0, 1])
        # as we moved robot, need to update target and obs
        # for rl policy
        return env, env.observation(env.get_obs()), no_target

    # RL Policy
    def learn(self, total_timesteps=10000, log_interval=100,
              max_episode_length=None, n_eval_ep=5):
        if not isinstance(total_timesteps, int):   # auto
            total_timesteps = int(total_timesteps)
        episode = 1
        s = self.env.reset()
        episode_return, episode_length = 0, 0
        best_return, best_eval_return = -np.inf, -np.inf
        most_tasks = 0
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
        self.env, s, _ = self.correct_position(self.env, s)
        for t in range(1, total_timesteps+1):
            s, done, success, episode_return, episode_length, plot_data, info = \
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
                                        plot_data, most_tasks,
                                        best_eval_return,
                                        n_eval_ep, max_episode_length)

            if(end_ep):
                best_return = \
                    self._on_train_ep_end(self.writer, t, episode,
                                          total_timesteps, best_return,
                                          episode_length, episode_return,
                                          success)
                # Reset everything
                episode += 1
                episode_return, episode_length = 0, 0
                s = self.env.reset()
                self.env, s, _ = self.correct_position(self.env, s)

        # Evaluate at end of training
        best_eval_return, plot_data = \
            self._eval_and_log(self.writer, t, episode,
                               plot_data, most_tasks, best_eval_return,
                               n_eval_ep, max_episode_length)

    def evaluate(self, env, max_episode_length=150, n_episodes=5,
                 print_all_episodes=False, render=False, save_images=False):
        stats = EpisodeStats(episode_lengths=[],
                             episode_rewards=[],
                             validation_reward=[])
        tasks, task_id = [], 0
        if(env.task == "pickup"):
            tasks = list(self.env.objects.keys())
            tasks.remove("table")
            tasks.remove("bin")
            n_episodes = len(tasks)

        ep_success = []
        # One episode per task
        for episode in range(n_episodes):
            s = env.reset()
            if(env.task == "pickup"):
                env.unwrapped.target = tasks[task_id]
                task_id += 1
            episode_length, episode_return = 0, 0
            done = False
            # Correct Position
            env, s, _ = self.correct_position(env, s)
            while(episode_length < max_episode_length // 2 and not done):
                # sample action and scale it to action space
                a, _ = self._pi.act(tt(s), deterministic=True)
                a = a.cpu().detach().numpy()
                ns, r, done, info = env.step(a)
                if(save_images):
                    # img, _ = self.find_target_center(env)
                    img = env.render()
                    self.im_lst.append(img)
                s = ns
                episode_return += r
                episode_length += 1
            if(episode_return >= 200 and env.task == "pickup"):
                self.move_to_box(env)
                success = self.eval_grasp_success(env)
            # Episode ended because it finished the task
            elif(env.task != "pickup" and r == 0):
                success = True
            else:
                success = False
            ep_success.append(success)
            stats.episode_rewards.append(episode_return)
            stats.episode_lengths.append(episode_length)
            if(print_all_episodes):
                print("Episode %d, Return: %.3f, Success: %s"
                      % (episode, episode_return, str(success)))

        # Save images
        if(save_images):
            if(not os.path.exists('./frames/')):
                os.makedirs("./frames/")
            for idx, im in enumerate(self.im_lst):
                cv2.imwrite("./frames/image_%04d.jpg" % idx, im)
        # mean and print
        mean_reward = np.mean(stats.episode_rewards)
        reward_std = np.std(stats.episode_rewards)
        mean_length = np.mean(stats.episode_lengths)
        length_std = np.std(stats.episode_lengths)

        self.log.info(
            "Success: %d/%d " % (np.sum(ep_success), len(ep_success)) +
            "Mean return: %.3f +/- %.3f, " % (mean_reward, reward_std) +
            "Mean length: %.3f +/- %.3f, over %d episodes" %
            (mean_length, length_std, n_episodes))
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
        # Set total timeout to timeout per task times all tasks + 1
        while(total_ts <= max_episode_length // 2 * n_tasks
              and self.no_detected_target < 3
              and not self.all_objs_in_box(env)):
            episode_length, episode_return = 0, 0
            done = False
            # Search affordances and correct position:
            env, s, no_target = self.correct_position(env, self.env.get_obs())
            if(no_target):
                # If no target model will move to initial position.
                # Search affordance from this position again
                self.correct_position(env, self.env.get_obs())

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
                # if(self.env.save_images):
                #     im = env.render()
                #     # self.im_lst.append(im)
                #     cv2.imwrite("./frames/image_%04d.jpg" % self.global_obs_it, im)
                #     self.global_obs_it += 1
            if(episode_return >= 200 and env.task == "pickup"):
                self.move_to_box(env)
                success = self.eval_grasp_success(env)
            else:
                success = False
            ep_success.append(success)
        self.save_images(env)
        self.log.info("Success: %d/%d " % (np.sum(ep_success), len(ep_success)))
        return ep_success

    def all_objs_in_box(self, env):
        for obj_name, obj in env.table_objs.items():
            if(not env.obj_in_box(obj)):
                return False
        return True

    def eval_grasp_success(self, env):
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
