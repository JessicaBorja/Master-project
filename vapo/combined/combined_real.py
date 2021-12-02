import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import sys
import math
from vapo.sac_agent.sac import SAC
from vapo.sac_agent.sac_utils.utils import tt

from vapo.affordance_model.datasets import get_transforms
from vapo.combined.target_search import TargetSearch


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
    def __init__(self, cfg, sac_cfg=None, wandb_login=None,
                 rand_target=False, *args, **kwargs):
        super(Combined, self).__init__(**sac_cfg, wandb_login=wandb_login)
        _aff_transforms = get_transforms(
            cfg.affordance.transforms.validation,
            cfg.target_search_aff.img_size)

        # initial angle
        _initial_obs = self.env.reset()["robot_obs"]
        self.origin = _initial_obs[:3]
        self.target_orn = np.array([math.pi, 0, 0])

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
        self.env.curr_detected_obj = self.target_pos
        self.env.unwrapped.curr_detected_obj = self.target_pos
        self.eval_env = self.env
        self.offset = cfg.env_wrapper.move_offset
        self.sim = False ##Not simulation

    def detect_and_correct(self, env):
        env.reset()
        # Compute target in case it moved
        # Area center is the target position + 5cm in z direction
        target_pos, no_target = self.target_search.compute(env)
        if(no_target):
            self.no_detected_target += 1
            input("No object detected. Please rearrange table.")
            return self.detect_and_correct(env)

        robot_target_pos = target_pos.copy()
        target_orn = self.target_orn.copy()
        # target_orn[2] += np.random.uniform(-1, 1) * np.radians(30)
        obs = self.env.reset(robot_target_pos,
                             target_orn,
                             self.offset)
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
        self.env, s, _ = self.detect_and_correct(self.env)

        # fps = FpsController(20)

        for ts in range(1, total_timesteps+1):
            # t = time.time()
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
                self.best_eval_return, self.most_tasks = \
                    self._eval_and_log(self.curr_ts,
                                       self.episode,
                                       self.most_tasks,
                                       self.best_eval_return,
                                       n_eval_ep, max_episode_length)
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
                # Go to origin then look for obj
                self.env, s, no_target = self.detect_and_correct(self.env)

            # fps.step()
            # print(1 / (time.time() - t))
            self.curr_ts += 1

    def evaluate(self, env, max_episode_length=150, n_episodes=5,
                 deterministic=True, **args):
        self.log.info("STARTING EVALUATION")
        ep_returns, ep_lengths = [], []
        no_target = True
        while(no_target):
            s = env.reset()
            target_pos, no_target, center_targets = \
                self.target_search.compute(env,
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
            s = env.reset(target_pos,
                          self.target_orn,
                          self.offset)
            while(episode_length < max_episode_length and not done):
                # sample action and scale it to action space
                a, _ = self._pi.act(tt(s), deterministic=deterministic)
                a = a.cpu().detach().numpy()
                ns, r, done, info = env.step(a)
                s = ns
                episode_return += r
                episode_length += 1
                time.sleep(0.05)
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
    def tidy_up(self, env, max_episode_length=100,
                n_objects=4, deterministic=True):
        self.target_search.random_target = False
        ep_success = []
        total_ts = 0
        s = env.reset()
        self.no_detected_target = 0
        # Set total timeout to timeout per task times all tasks + 1
        while(total_ts <= max_episode_length * n_objects):
            episode_length, episode_return = 0, 0
            done = False
            env, s, no_target = self.detect_and_correct(env)

            # If it did not find a target again, terminate everything
            while(episode_length < max_episode_length
                  and not done):
                # sample action and scale it to action space
                a, _ = self._pi.act(tt(s),
                                    deterministic=deterministic)
                a = a.cpu().detach().numpy()
                ns, r, done, info = env.step(a, move_to_box=True)
                s = ns
                episode_return += r
                episode_length += 1
                total_ts += 1
                success = info['success']
            self.log.info(
                "Success: %s " % str(success) +
                "Return: %.3f" % episode_return)
            ep_success.append(success)
        self.log.info("Success: %d/%d " % (np.sum(ep_success), len(ep_success)))
        return ep_success
