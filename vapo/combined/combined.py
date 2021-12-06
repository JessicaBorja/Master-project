import numpy as np
import math
import sys

from vapo.sac_agent.sac import SAC
from vapo.sac_agent.sac_utils.utils import tt
from vapo.affordance_model.datasets import get_transforms
from vapo.combined.target_search import TargetSearch


class Combined(SAC):
    def __init__(self, cfg, sac_cfg=None, wandb_login=None):
        super(Combined, self).__init__(**sac_cfg, wandb_login=wandb_login)
        _cam_id = self._find_cam_id()
        _aff_transforms = get_transforms(
            cfg.affordance.transforms.validation,
            cfg.target_search.aff_cfg.img_size)

        # initial pose
        self.origin = self.env.origin
        self.target_orn = self.env.target_orn

        # To enumerate static cam preds on target search
        self.no_detected_target = 0

        args = {"cam_id": _cam_id,
                "initial_pos": self.origin,
                "aff_transforms": _aff_transforms,
                **cfg.target_search}
        _class_label = self.get_task_label()
        self.target_search = TargetSearch(self.env,
                                          class_label=_class_label,
                                          **args)
        self.p_dist = None
        if(self.env.task == "pickup"):
            if(self.env.rand_positions):
                self.p_dist = \
                    {c: 0 for c in self.env.scene.objs_per_class.keys()}

        self.target_pos, _ = self.target_search.compute(rand_sample=True)

        # Target specifics
        self.env.target_search = self.target_search
        self.env.curr_detected_obj = self.target_pos
        self.env.unwrapped.curr_detected_obj = self.target_pos
        self.eval_env = self.env
        self.radius = self.env.termination_radius  # Distance in meters
        self.move_above = 0.03
        self.sim = True

    def get_task_label(self):
        task = self.env.get_task()
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
    def detect_and_correct(self, env, obs,
                           p_dist=None, noisy=False,
                           rand_sample=True):
        if(obs is None):
            obs = env.reset()
        # Compute target in case it moved
        # Area center is the target position + 5cm in z direction
        target_pos, no_target = \
            self.target_search.compute(env,
                                       p_dist=p_dist,
                                       noisy=noisy,
                                       rand_sample=rand_sample)
        if(no_target):
            self.no_detected_target += 1
        res = self.correct_position(env, obs, target_pos, no_target)
        return res

    def correct_position(self, env, s, target_pos, no_target):
        # Take current robot state
        if(isinstance(s, dict)):
            s = s["robot_obs"]
        tcp_pos = s[:3]

        # Set current_target in each episode
        self.target_pos = target_pos
        env.curr_detected_obj = target_pos

        # Move
        if(np.linalg.norm(tcp_pos - target_pos) > self.radius):
            if(env.task == "pickup" or env.task == "drawer"):
                # To never collide with the box
                z_value = max(self.target_pos[2] + 0.09, 0.8)
                up_target = [tcp_pos[0],
                             tcp_pos[1],
                             z_value]
                # Move up from starting pose
                a = [up_target, self.target_orn, 1]
                tcp_pos = env.move_to_target(tcp_pos, a)

                # Move to target
                reach_target = [*self.target_pos[:2], tcp_pos[-1]]
                a = [reach_target, self.target_orn, 1]
                tcp_pos = env.move_to_target(tcp_pos, a)
                if(self.target_search.mode == "env"):
                    # Environment returns the center of mass..
                    move_to = \
                        self.target_pos + np.array([0, 0, self.move_above])
                else:
                    # Affordances detect the surface of an object
                    if(env.task == "pickup"):
                        move_to = \
                            self.target_pos + np.array(
                                [0, 0, self.move_above - 0.01])
                    else:
                        move_to = \
                            self.target_pos + np.array([0.03, 0.02, 0.05])
            else:
                # Move in x-z dir
                x_target = [self.target_pos[0], tcp_pos[1], self.target_pos[2]]
                a = [x_target, self.target_orn, 1]
                tcp_pos = env.move_to_target(tcp_pos, a)

                # Move up y dir
                move_to = self.target_pos

            # Move to target
            a = [move_to, self.target_orn, 1]
            tcp_pos = env.move_to_target(tcp_pos, a)
        # as we moved robot, need to update target and obs
        # for rl policy
        return env, env.observation(env.get_obs()), no_target

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
        self.env, s, _ = self.detect_and_correct(self.env, None,
                                                 self.p_dist,
                                                 noisy=True)
        for ts in range(1, total_timesteps+1):
            s, done, success, episode_return, episode_length, plot_data, info = \
                self.training_step(s, self.curr_ts, episode_return, episode_length)

            # End episode
            timeout = (max_episode_length
                       and (episode_length >= max_episode_length))
            end_ep = timeout or done

            # Log interval (sac)
            if((ts % log_interval == 0 and not self._log_by_episodes)
               or (self._log_by_episodes and end_ep
                   and self.episode % _log_n_ep == 0)):
                eval_all_objs = self.episode % (2 * _log_n_ep) == 0
                self.best_eval_return, self.most_tasks = \
                    self._eval_and_log(self.curr_ts,
                                       self.episode,
                                       self.most_tasks,
                                       self.best_eval_return,
                                       n_eval_ep, max_episode_length)
                if(self.eval_env.rand_positions and eval_all_objs):
                    _, self.most_full_tasks = \
                        self._eval_and_log(self.curr_ts,
                                           self.episode,
                                           self.most_full_tasks,
                                           self.best_eval_return,
                                           n_eval_ep, max_episode_length,
                                           eval_all_objs=True)
            if end_ep:
                self.best_return = \
                    self._on_train_ep_end(self.curr_ts,
                                          self.episode,
                                          total_timesteps,
                                          self.best_return,
                                          episode_length, episode_return,
                                          success,
                                          plot_data)
                # Reset everything
                self.episode += 1
                self.env.obs_it = 0
                episode_return, episode_length = 0, 0
                self.env, s, _ = self.detect_and_correct(self.env, None,
                                                         self.p_dist,
                                                         noisy=True)
            self.curr_ts += 1
        # Evaluate at end of training
        for eval_all_objs in [False, True]:
            if(eval_all_objs and self.env.rand_positions
               or not eval_all_objs):
                self.best_eval_return, most_tasks = \
                    self._eval_and_log(self.curr_ts,
                                       self.episode,
                                       self.most_tasks,
                                       self.best_eval_return,
                                       n_eval_ep, max_episode_length,
                                       eval_all_objs=eval_all_objs)

    # Only applies to pickup task
    def eval_all_objs(self, env, max_episode_length,
                      n_episodes=None, print_all_episodes=False,
                      render=False, save_images=False):
        if(env.rand_positions is None):
            return

        # Store current objs to restore them later
        previous_objs = env.scene.table_objs

        #
        tasks = env.scene.obj_names
        n_objs = len(env.scene.rand_positions)
        n_total_objs = len(tasks)
        succesful_objs = []
        episodes_success = []
        for i in range(math.ceil(n_total_objs/n_objs)):
            if(len(tasks[i:]) >= n_objs):
                curr_objs = tasks[i*n_objs: i*n_objs + n_objs]
            else:
                curr_objs = tasks[i:]
            env.load_scene_with_objects(curr_objs)
            mean_reward, mean_length, ep_success, success_objs = \
                self.evaluate(env,
                              max_episode_length=max_episode_length,
                              render=render,
                              save_images=save_images)
            succesful_objs.extend(success_objs)
            episodes_success.extend(ep_success)

        self.log.info(
            "Full evaluation over %d objs \n" % n_total_objs +
            "Success: %d/%d " % (np.sum(episodes_success), len(ep_success)))

        # Restore scene before loading full sweep eval
        env.load_scene_with_objects(previous_objs)
        return episodes_success, succesful_objs

    def evaluate(self, env, max_episode_length=150, n_episodes=5,
                 print_all_episodes=False, render=False, save_images=False):
        ep_returns, ep_lengths = [], []
        tasks, task_it = [], 0

        if(env.task == "pickup"):
            if(self.target_search.mode == "env"):
                tasks = env.scene.table_objs
                n_episodes = len(tasks)
            else:
                # Search by affordance
                s = env.reset()
                target_pos, no_target, center_targets = \
                    self.target_search.compute(env,
                                               return_all_centers=True)
                n_episodes = len(center_targets)

        ep_success = []
        success_objs = []
        # One episode per task
        for episode in range(n_episodes):
            s = env.reset(eval=True)
            if(env.task == "pickup"):
                if(self.target_search.mode == "env"):
                    env.set_target(tasks[task_it])
                    target_pos, no_target = \
                        self.target_search.compute(env)
                else:
                    target_pos = center_targets[task_it]["target_pos"]
                    env.set_target(center_targets[task_it]["target_str"])
                task_it += 1
            else:
                target_pos, no_target = \
                        self.target_search.compute(env)
            episode_length, episode_return = 0, 0
            done = False
            # Correct Position
            env, s, _ = self.correct_position(env, s, target_pos, no_target)
            while(episode_length < max_episode_length // 2 and not done):
                # sample action and scale it to action space
                a, _ = self._pi.act(tt(s), deterministic=True)
                a = a.cpu().detach().numpy()
                ns, r, done, info = env.step(a)
                s = ns
                episode_return += r
                episode_length += 1
            if(episode_return >= 200 and env.task == "pickup"):
                env.move_to_box()
                success = info["success"]
                if(success):
                    success_objs.append(env.target)
            # Episode ended because it finished the task
            elif(env.task != "pickup" and r > 0):
                success = True
            else:
                success = False
            ep_success.append(success)
            ep_returns.append(episode_return)
            ep_lengths.append(episode_length)
            if(print_all_episodes):
                print("Episode %d, Return: %.3f, Success: %s"
                      % (episode, episode_return, str(success)))

        # mean and print
        mean_reward, reward_std = np.mean(ep_returns), np.std(ep_returns)
        mean_length, length_std = np.mean(ep_lengths), np.std(ep_lengths)

        self.log.info(
            "Success: %d/%d " % (np.sum(ep_success), len(ep_success)) +
            "Mean return: %.3f +/- %.3f, " % (mean_reward, reward_std) +
            "Mean length: %.3f +/- %.3f, over %d episodes" %
            (mean_length, length_std, n_episodes))
        return mean_reward, mean_length, ep_success, success_objs

    # Only applies to tabletop
    def tidy_up(self, env, max_episode_length=100):
        tasks = []
        # get from static cam affordance
        if(env.task == "pickup"):
            tasks = self.env.scene.table_objs
            n_tasks = len(tasks)

        ep_success = []
        total_ts = 0
        s = env.reset()
        # Set total timeout to timeout per task times all tasks + 1
        while(total_ts <= max_episode_length * n_tasks
              and self.no_detected_target < 3
              and not self.all_objs_in_box(env)):
            episode_length, episode_return = 0, 0
            done = False
            # Search affordances and correct position:
            env, s, no_target = self.detect_and_correct(env,
                                                        self.env.get_obs(),
                                                        rand_sample=False)
            if(no_target):
                # If no target model will move to initial position.
                # Search affordance from this position again
                env, s, no_target = \
                    self.detect_and_correct(env, self.env.get_obs(),
                                            rand_sample=False)

            # If it did not find a target again, terminate everything
            init_pos = s['robot_obs'][:3]
            dist = 0
            while(episode_length < max_episode_length
                  and self.no_detected_target < 3
                  and dist < env.termination_radius
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
                dist = np.linalg.norm(init_pos - s['robot_obs'][:3])
            if(episode_return >= 200 and env.task == "pickup"):
                env.move_to_box(env, sample=True)
                success = info["success"]
            else:
                success = False
            ep_success.append(success)
        self.log.info(
            "Success: %d/%d " % (np.sum(ep_success), len(ep_success)))
        return ep_success