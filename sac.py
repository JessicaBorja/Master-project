from datetime import datetime
import os, sys
import json, pickle
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal
import itertools

import time
from utils.replay_buffer import ReplayBuffer
from utils.utils import EpisodeStats, tt, soft_update
from utils.networks import ActorNetwork, CriticNetwork, ValueNetwork

class SAC():
    def __init__(self, env, eval_env= None, gamma = 0.99, alpha = "auto" , \
                 actor_lr = 1e-5, critic_lr = 1e-5, alpha_lr = 1e-5, hidden_dim = 256,
                 tau = 0.005, train_freq = 1, gradient_steps = 1, learning_starts = 1000,\
                 target_update_interval = 1, batch_size = 256, buffer_size = 1e6, model_name = "sac"):
        self.env = env
        self.eval_env = eval_env
        #Replay buffer
        self._max_size = buffer_size
        self._replay_buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size

        #Agent
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self._gamma = gamma
        self.tau = tau
        self.target_update_interval = target_update_interval

        #networks
        self._auto_entropy = False
        if isinstance(alpha, str): #auto
            self._auto_entropy = True
            self.ent_coef = 1 #entropy coeficient
            self.target_entropy = -np.prod(env.action_space.shape).item()  # heuristic value
            self.log_ent_coef = torch.zeros(1, requires_grad=True, device="cuda") #init value
            self.ent_coef_optimizer = optim.Adam([self.log_ent_coef], lr = alpha_lr)
        else:
            self.ent_coef = alpha #entropy coeficient

        self.learning_starts = learning_starts

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_max = env.action_space.high[0]
        self._pi = ActorNetwork(state_dim, action_dim, action_max=action_max, hidden_dim = hidden_dim).cuda()
        self._pi_optim = optim.Adam(self._pi.parameters(), lr = actor_lr)

        self._q1 = CriticNetwork(state_dim, action_dim,hidden_dim = hidden_dim).cuda()
        self._q1_target = CriticNetwork(state_dim, action_dim,hidden_dim = hidden_dim ).cuda()
        self._q1_target.load_state_dict(self._q1.state_dict())
        self._q1_optimizer = optim.Adam(self._q1.parameters(), lr = critic_lr)
        
        self._q2 = CriticNetwork(state_dim, action_dim,hidden_dim = hidden_dim).cuda()
        self._q2_target = CriticNetwork(state_dim, action_dim, hidden_dim = hidden_dim ).cuda()
        self._q2_target.load_state_dict(self._q2.state_dict())
        self._q2_optimizer = optim.Adam(self._q2.parameters(), lr = critic_lr)
        
        _q_params =  itertools.chain(self._q1.parameters(), self._q2.parameters())
        self._q_optim = optim.Adam(_q_params, lr = critic_lr)
        self._loss_function = nn.MSELoss()
        #Summary Writer
        if not os.path.exists("./results"):
            os.makedirs("./results")
        self.model_name = "{}_{}".format(model_name, datetime.now().strftime('%d-%m_%I-%M'))
        self.writer_name = "./results/{}".format(self.model_name)
        #self.eval_writer_name = "./results/%s_eval"%self.model_name
        self.trained_path = "./trained_models/{}".format(self.model_name)
    
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
    def learn(self, total_timesteps = 10000, log_interval=100 , max_episode_length = None):
        stats = EpisodeStats(episode_lengths = [], episode_rewards = [])
        #eval_writer = SummaryWriter(self.eval_writer_name)
        writer = SummaryWriter(self.writer_name)
        episode = 0
        s = self.env.reset()
        episode_reward, episode_length, avg_reward = 0,0,0 #avg_reward over log_interval timesteps
        best_reward, best_eval_reward = -np.inf, -np.inf
        if(max_episode_length is None):
            max_episode_length = sys.maxsize #"infinite"
        
        losses = {"actor_loss": [] , "critic_loss": [], "ent_coef_loss": []}
        for t in range(1, total_timesteps+1):
            a, _ = self._pi.act(tt(s), deterministic = False)#sample action and scale it to action space
            a = a.cpu().detach().numpy()
            ns, r, done, _ = self.env.step(a)
            self._replay_buffer.add_transition(s, a, r, ns, done)
            s = ns
            avg_reward+=r
            episode_reward +=r
            episode_length +=1                   
            
            #Replay buffer has enough data
            if(self._replay_buffer.__len__()>= self.batch_size and not done and t>self.learning_starts):
                sample = self._replay_buffer.sample(self.batch_size)
                batch_states, batch_actions, batch_rewards, batch_next_states, batch_terminal_flags = sample

                if(t % self.train_freq == 0): #If we have done train_freq env. steps
                    for n in range(self.gradient_steps):
                        #---------------- Critic networks update -------------#
                        with torch.no_grad():
                            next_actions, log_probs = self._pi.act(batch_next_states, deterministic=False, reparametrize = False)
                            #next_actions *= self.env.action_space.high[0] #scale action between high and -high action space

                            target_qvalue = torch.min(
                                self._q1_target(batch_next_states, next_actions),
                                self._q2_target(batch_next_states, next_actions) )
                            
                            td_target = batch_rewards + (1 - batch_terminal_flags) * self._gamma * \
                                    (target_qvalue - self.ent_coef * log_probs)     
                        #Critic 1
                        curr_prediction_c1 = self._q1(batch_states, batch_actions)                
                        #self._q1_optimizer.zero_grad()
                        loss_c1 = self._loss_function(curr_prediction_c1, td_target.detach())
                        #loss_c1.backward()
                        #nn.utils.clip_grad_norm_(self._q1.parameters(),1)
                        #self._q1_optimizer.step()
                        
                        #Critic 2
                        curr_prediction_c2 = self._q2(batch_states, batch_actions)
                        #self._q2_optimizer.zero_grad()
                        loss_c2 = self._loss_function(curr_prediction_c2, td_target.detach())
                        #loss_c2.backward()
                        #nn.utils.clip_grad_norm_(self._q2.parameters(),1)
                        #self._q2_optimizer.step()
                        #--- update two critics w/same optimizer ---#
                        self._q_optim.zero_grad()
                        loss_critics = loss_c1 + loss_c2
                        loss_critics.backward()
                        self._q_optim.step()

                        losses["critic_loss"] += [loss_c1.item(), loss_c2.item()]
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
                        losses["actor_loss"].append(policy_loss.item())

                        #---------------- Entropy network update -------------#
                        ent_coef_loss = self.update_entropy(log_probs)
                        losses["ent_coef_loss"].append(ent_coef_loss)
                        #losses["ent_coed"].append(self.ent_coef)
                        
                        #------------------ Target Networks update -------------------#
                        soft_update(self._q1_target, self._q1, self.tau)
                        soft_update(self._q2_target, self._q2, self.tau)

            if(done or (max_episode_length and (episode_length >= max_episode_length))): #End episode
                stats.episode_lengths.append(episode_length)
                stats.episode_rewards.append(episode_reward)
                print("Episode %d: %d Steps, Reward: %.3f, total timesteps: %d/%d "%(episode, episode_length, episode_reward, t, total_timesteps))
                #Summary Writer
                writer.add_scalar('train/episode_reward', episode_reward, episode)
                writer.add_scalar('train/episode_length', episode_length, episode)

                if(episode_reward>best_reward): #and episode>20):
                    # if(episode_reward<0):
                    #     self.save(self.trained_path+"_r-neg%d.pth"%(np.abs(round(episode_reward))))
                    # else:
                    #     self.save(self.trained_path+"_r-%d.pth"%(round(episode_reward)))
                    print("New best train ep. reward!%.3f"%episode_reward)
                    self.save(self.trained_path+"_best.pth")
                    best_reward = episode_reward
                
                #Reset everything
                episode+=1
                episode_reward, episode_length = 0,0
                s = self.env.reset()

            if(t%log_interval==0):
                for key,value in losses.items():
                    if value: #not empty
                        mean_loss = np.mean(value)
                        writer.add_scalar("train/%s"%key, mean_loss, t)
                
                losses = {"actor_loss": [] , "critic_loss": [], "ent_coef_loss": []}
                if(self.eval_env is not None):
                    mean_reward, mean_length = self.evaluate(self.eval_env, max_episode_length, model_name = self.model_name)
                    if(mean_reward > best_eval_reward):
                        print("New best eval reward!%.3f"%mean_reward)
                        self.save(self.trained_path+"_best_eval.pth")
                        best_eval_reward = mean_reward
                    writer.add_scalar('eval/mean_reward', mean_reward, t)
                    writer.add_scalar('eval/mean_ep_length', mean_length, t)
                    # self.log_tensorboard_eval(self.eval_env, writer, step = t,\
                    #                     max_episode_length=max_episode_length, model_name=self.model_name) #timesteps in x axis
                avg_reward = 0

        if(self.eval_env is not None):
            print("End of training evaluation:")
            self.evaluate(self.eval_env, model_name = self.model_name, print_all_episodes = True)
        #self.save_stats(stats, self.writer_name)
        return stats

    def log_tensorboard_eval(self, env, writer, step, max_episode_length, model_name="sac", write_file = False):
        mean_reward, mean_length = self.evaluate(env, max_episode_length=max_episode_length,\
                                                 model_name = model_name, write_file = write_file)
        writer.add_scalar('eval/mean_reward', mean_reward, step)
        writer.add_scalar('eval/mean_ep_length', mean_length, step)

    def evaluate(self, env, max_episode_length = 150, n_episodes = 10, model_name = "sac",\
                 print_all_episodes = False, write_file = False, render = False):
        stats = EpisodeStats(episode_lengths = [], episode_rewards = [])
        for episode in range(n_episodes):
            s = env.reset()
            episode_length, episode_reward = 0,0
            done = False
            while( episode_length < max_episode_length and not done):
                a, _ = self._pi.act(tt(s), deterministic = True)#sample action and scale it to action space
                a = a.cpu().detach().numpy()
                ns, r, done, _ = env.step(a)
                if(render):
                    env.render()
                s = ns
                episode_reward+=r
                episode_length+=1
            stats.episode_rewards.append(episode_reward)
            stats.episode_lengths.append(episode_length)
            if(print_all_episodes):
                print("Episode %d, Reward: %.3f"%(episode, episode_reward))
        #mean and print
        mean_reward = np.mean(stats.episode_rewards)
        reward_std = np.std(stats.episode_rewards)
        mean_length =  np.mean(stats.episode_lengths)
        length_std = np.std(stats.episode_lengths)

        #write_file
        if(write_file):
            file_name = "./results/%s_eval.json"%model_name
            with open(file_name, 'w') as fp:
                itemlist = {"episode_rewards: ": stats.episode_rewards,
                            "episode_lengths: ": stats.episode_lengths}
                json.dump(itemlist, fp)
            print("Evaluation results file at: %s"%os.path.abspath(file_name))

        print("Mean reward: %.3f +/- %.3f , Mean length: %.3f +/- %.3f, over %d episodes"%(mean_reward, reward_std, mean_length, length_std, n_episodes))
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
