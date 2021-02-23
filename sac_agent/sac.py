from datetime import datetime
import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import itertools
from sac_agent.sac_utils.replay_buffer import ReplayBuffer
from sac_agent.sac_utils.utils import EpisodeStats, tt, soft_update, get_nets
import logging
# A logger for this file
log = logging.getLogger(__name__)


class SAC():
    def __init__(self, env=None, eval_env=None, save_dir="./trained_models",
                 gamma=0.99, alpha="auto",
                 actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4,
                 tau=0.005, learning_starts=1000, img_obs=False,
                 batch_size=256, buffer_size=1e6,
                 model_name="sac", net_cfg=None):
        self.save_dir = save_dir
        self.env = env
        self.eval_env = eval_env
        # Replay buffer
        self._max_size = buffer_size
        self._replay_buffer = ReplayBuffer(buffer_size, img_obs)
        self.batch_size = batch_size

        # Agent
        self._gamma = gamma
        self.tau = tau

        #networks
        self._auto_entropy = False
        if isinstance(alpha, str): #auto
            self._auto_entropy = True
            self.ent_coef = 1 #entropy coeficient
            self.target_entropy = -np.prod(env.action_space.shape).item()  # heuristic value
            self.log_ent_coef = torch.zeros(1, requires_grad=True, device="cuda") #init value
            self.ent_coef_optimizer = optim.Adam([self.log_ent_coef], lr = alpha_lr)
        else:
            self.ent_coef = alpha # entropy coeficient

        self.learning_starts = learning_starts

        obs_space = env.observation_space
        action_dim = env.action_space.shape[0]
        action_max = env.action_space.high[0]

        policy_net, critic_net, obs_space = get_nets(img_obs, obs_space)
        self._pi = policy_net(obs_space, action_dim, action_max=action_max, **net_cfg).cuda()
        self._q1 = critic_net(obs_space, action_dim, **net_cfg).cuda()
        self._q1_target = critic_net(obs_space, action_dim, **net_cfg).cuda()
        self._q2 = critic_net(obs_space, action_dim,**net_cfg).cuda()
        self._q2_target = critic_net(obs_space, action_dim, **net_cfg).cuda()
        
        self._pi_optim = optim.Adam(self._pi.parameters(), lr=actor_lr)

        self._q1_target.load_state_dict(self._q1.state_dict())
        self._q1_optimizer = optim.Adam(self._q1.parameters(), lr = critic_lr)
        
        self._q2_target.load_state_dict(self._q2.state_dict())
        self._q2_optimizer = optim.Adam(self._q2.parameters(), lr = critic_lr)
        
        _q_params =  itertools.chain(self._q1.parameters(), self._q2.parameters())
        self._q_optim = optim.Adam(_q_params, lr = critic_lr)
        self._loss_function = nn.MSELoss()
        #Summary Writer
        if not os.path.exists("./results"):
            os.makedirs("./results")
        #models folder
        if not os.path.exists(save_dir): 
            os.makedirs(save_dir)
        self.model_name = "{}_{}".format(model_name, datetime.now().strftime('%d-%m_%H-%M'))
        self.writer_name = "./results/{}".format(self.model_name)
        #self.eval_writer_name = "./results/%s_eval"%self.model_name
        self.trained_path = "{}/{}".format(self.save_dir, self.model_name)
    
    def update_entropy(self, log_probs):
        if( self._auto_entropy ):
            self.ent_coef_optimizer.zero_grad()
            ent_coef_loss = -(self.log_ent_coef * (log_probs + self.target_entropy).detach()).mean()
            ent_coef_loss.backward()
            self.ent_coef_optimizer.step()
            self.ent_coef = self.log_ent_coef.exp()
            return ent_coef_loss.item()
        else:
            return 0
    
    def update(self, td_target, batch_states, batch_actions, plot_data):
        #Critic 1
        curr_prediction_c1 = self._q1(batch_states, batch_actions)                
        loss_c1 = self._loss_function(curr_prediction_c1, td_target.detach())

        #Critic 2
        curr_prediction_c2 = self._q2(batch_states, batch_actions)
        loss_c2 = self._loss_function(curr_prediction_c2, td_target.detach())
        #--- update two critics w/same optimizer ---#
        self._q_optim.zero_grad()
        loss_critics = loss_c1 + loss_c2
        loss_critics.backward()
        self._q_optim.step()

        plot_data["critic_loss"] += [loss_c1.item(), loss_c2.item()]
        #---------------- Policy network update -------------#
        predicted_actions, log_probs = self._pi.act(batch_states, deterministic = False, reparametrize = True)
        critic_value = torch.min(
            self._q1(batch_states, predicted_actions),
            self._q2(batch_states, predicted_actions) )
        #Actor update/ gradient ascent
        self._pi_optim.zero_grad()
        policy_loss = (self.ent_coef * log_probs - critic_value).mean()
        policy_loss.backward()
        self._pi_optim.step()
        plot_data["actor_loss"].append(policy_loss.item())

        #---------------- Entropy network update -------------#
        ent_coef_loss = self.update_entropy(log_probs)
        plot_data["ent_coef"].append(self.ent_coef)
        plot_data["ent_coef_loss"].append(ent_coef_loss)
        
        #------------------ Target Networks update -------------------#
        soft_update(self._q1_target, self._q1, self.tau)
        soft_update(self._q2_target, self._q2, self.tau)
        
        return plot_data

    def learn(self, total_timesteps = 10000, log_interval=100 , max_episode_length = None, n_eval_ep=5):
        if not isinstance(total_timesteps, int): #auto
            total_timesteps =  int(total_timesteps)
        stats = EpisodeStats(episode_lengths = [], episode_rewards = [], validation_reward = [])
        #eval_writer = SummaryWriter(self.eval_writer_name)
        writer = SummaryWriter(self.writer_name)
        episode = 0
        s = self.env.reset()
        episode_reward, episode_length = 0,0
        best_reward, best_eval_reward = -np.inf, -np.inf
        if(max_episode_length is None):
            max_episode_length = sys.maxsize #"infinite"
        
        plot_data = {"actor_loss": [] , "critic_loss": [], "ent_coef_loss": [], "ent_coef":[]}
        for t in range(1, total_timesteps+1):
            a, _ = self._pi.act(tt(s), deterministic = False)#sample action and scale it to action space
            a = a.cpu().detach().numpy()
            ns, r, done, info = self.env.step(a)

            self._replay_buffer.add_transition(s, a, r, ns, done)
            s = ns
            episode_reward +=r
            episode_length +=1                   
            
            #Replay buffer has enough data
            if(self._replay_buffer.__len__()>= self.batch_size and not done and t>self.learning_starts):
                sample = self._replay_buffer.sample(self.batch_size)
                batch_states, batch_actions, batch_rewards, batch_next_states, batch_terminal_flags = sample

                with torch.no_grad():
                    next_actions, log_probs = self._pi.act(batch_next_states, deterministic=False, reparametrize = False)
                    #next_actions *= self.env.action_space.high[0] #scale action between high and -high action space

                    target_qvalue = torch.min(
                        self._q1_target(batch_next_states, next_actions),
                        self._q2_target(batch_next_states, next_actions) )
                    
                    td_target = batch_rewards + (1 - batch_terminal_flags) * self._gamma * \
                            (target_qvalue - self.ent_coef * log_probs)     
                    
                #----------------  Networks update -------------#
                plot_data = self.update(td_target, batch_states, batch_actions, plot_data)

            if(done or (max_episode_length and (episode_length >= max_episode_length))): #End episode
                stats.episode_lengths.append(episode_length)
                stats.episode_rewards.append(episode_reward)
                log.info("Episode %d: %d Steps, Return: %.3f, total timesteps: %d/%d "%(episode, episode_length, episode_reward, t, total_timesteps))
                #Summary Writer
                writer.add_scalar('train/episode_return', episode_reward, episode)
                writer.add_scalar('train/episode_length', episode_length, episode)

                if(episode_reward>best_reward):
                    log.info("[%d] New best train ep. return!%.3f"%(episode, episode_reward))
                    self.save(self.trained_path+"_best.pth")
                    best_reward = episode_reward
                
                #Reset everything
                episode+=1
                episode_reward, episode_length = 0,0
                s = self.env.reset()

            if(t%log_interval==0):
                for key,value in plot_data.items():
                    if value: #not empty
                        if(key == "critic_loss"):
                            data = np.mean(value[-1])
                        else:
                            data = value[-1]#np.mean(value)
                        writer.add_scalar("train/%s"%key, data, t)
                
                plot_data = {"actor_loss": [] , "critic_loss": [], "ent_coef_loss": [], "ent_coef":[]}
                if(self.eval_env is not None):
                    mean_reward, mean_length = self.evaluate(self.eval_env, max_episode_length,n_episodes=n_eval_ep)
                    stats.validation_reward.append(mean_reward)
                    if(mean_reward > best_eval_reward):
                        log.info("[%d] New best eval avg. return!%.3f"%(episode, mean_reward))
                        self.save(self.trained_path+"_best_eval.pth")
                        best_eval_reward = mean_reward
                    writer.add_scalar('eval/mean_return(%dep)'%(n_eval_ep), mean_reward, t)
                    writer.add_scalar('eval/mean_ep_length(%dep)'%(n_eval_ep), mean_length, t)
        if(self.eval_env is not None):
            log.info("End of training evaluation:")
            self.evaluate(self.eval_env, max_episode_length, print_all_episodes = True)
        return stats
        
    def evaluate(self, env, max_episode_length=150, n_episodes=5,\
                 print_all_episodes=False, render=False, save_images=False):
        stats = EpisodeStats(episode_lengths=[], episode_rewards=[], validation_reward=[])
        im_lst = []
        for episode in range(n_episodes):
            s = env.reset()
            episode_length, episode_reward = 0,0
            done = False
            while( episode_length < max_episode_length and not done):
                a, _ = self._pi.act(tt(s), deterministic = True)#sample action and scale it to action space
                a = a.cpu().detach().numpy()
                ns, r, done, info = env.step(a)
                if(render):
                    img = env.render()
                    im_lst.append(img)
                s = ns
                episode_reward+=r
                episode_length+=1
            stats.episode_rewards.append(episode_reward)
            stats.episode_lengths.append(episode_length)
            if(print_all_episodes):
                print("Episode %d, Return: %.3f"%(episode, episode_reward))
        
        # Save images
        if(save_images):
            import cv2
            os.makedirs("./frames/")
            for idx, im in enumerate(im_lst):
                cv2.imwrite("./frames/image_%04d.jpg"%idx, im)
        # mean and print
        mean_reward = np.mean(stats.episode_rewards)
        reward_std = np.std(stats.episode_rewards)
        mean_length =  np.mean(stats.episode_lengths)
        length_std = np.std(stats.episode_lengths)
        
        log.info("Mean return: %.3f +/- %.3f , Mean length: %.3f +/- %.3f, over %d episodes"%(mean_reward, reward_std, mean_length, length_std, n_episodes))
        print("Mean return: %.3f +/- %.3f , Mean length: %.3f +/- %.3f, over %d episodes"%(mean_reward, reward_std, mean_length, length_std, n_episodes))
        return mean_reward, mean_length

    def save(self, path):
        save_dict = {
            'actor_dict': self._pi.state_dict(),
            'actor_optimizer_dict': self._pi_optim.state_dict(),

            'critic_1_dict': self._q1.state_dict(),
            'critic_1_target_dict': self._q1_target.state_dict(),
            'critic_1_optimizer_dict': self._q1_optimizer.state_dict(),

            'critic_2_dict': self._q2.state_dict(),
            'critic_2_target_dict': self._q2_target.state_dict(),
            'critic_2_optimizer_dict': self._q2_optimizer.state_dict(),
            'critics_optim': self._q_optim.state_dict(),
            
            'ent_coef': self.ent_coef}
        if self._auto_entropy:
            save_dict['ent_coef_optimizer'] = self.ent_coef_optimizer.state_dict()
        torch.save(save_dict, path)
  
    def load(self, path):
        if os.path.isfile(path):
            print("Loading checkpoint")
            checkpoint = torch.load(path)

            self._pi.load_state_dict(checkpoint['actor_dict'])
            self._pi_optim.load_state_dict(checkpoint['actor_optimizer_dict'])

            self._q1.load_state_dict(checkpoint['critic_1_dict'])
            self._q1_target.load_state_dict(checkpoint['critic_1_target_dict'])
            self._q1_optimizer.load_state_dict(checkpoint['critic_1_optimizer_dict'])

            self._q2.load_state_dict(checkpoint['critic_2_dict'])
            self._q2_target.load_state_dict(checkpoint['critic_2_target_dict'])
            self._q2_optimizer.load_state_dict(checkpoint['critic_2_optimizer_dict'])
            # self.ent_coef =  checkpoint["ent_coef"]
            # self.ent_coef_optimizer.load_state_dict(checkpoint['ent_coef_optimizer_dict'])
            print("load done")
            return True
        else:
            print("no path " + path)
            return False

    def save_stats(self, stats, path):
        rewards_path = path + "_rewards.txt"
        with open(rewards_path, 'wb') as fp:
            pickle.dump(stats.episode_rewards, fp)

        lengths_path = path + "_ep_lengths.txt"    
        with open(lengths_path, 'wb') as fp:
            pickle.dump(stats.episode_lengths, fp)
