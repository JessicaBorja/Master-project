from datetime import datetime
import os
import json
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal
import time
from utils.replay_buffer import ReplayBuffer
from utils.utils import EpisodeStats, tt, soft_update
from utils.networks import ActorNetwork, CriticNetwork, ValueNetwork

class SAC():
    def __init__(self, env, eval_env= None, gamma = 0.99, learning_rate = 3e-4, ent_coef = 1 , \
                 tau = 0.005, train_freq = 1, gradient_steps = 1,\
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
        self.ent_coef = ent_coef #entropy coeficient
        self.target_entropy = -np.prod(env.action_space.shape).item()  # heuristic value
        self.log_ent_coef = torch.zeros(1, requires_grad=True, device="cuda")
        self.ent_coef_optimizer = optim.Adam([self.log_ent_coef], lr = learning_rate)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self._actor = ActorNetwork(state_dim, action_dim).cuda()
        self._actor_optimizer = optim.Adam(self._actor.parameters(), lr = learning_rate)

        self._critic_1 = CriticNetwork(state_dim, action_dim ).cuda()
        self._critic_1_target = CriticNetwork(state_dim, action_dim ).cuda()
        self._critic_1_target.load_state_dict(self._critic_1.state_dict())
        self._critic_1_optimizer = optim.Adam(self._critic_1.parameters(), lr = learning_rate)
        
        self._critic_2 = CriticNetwork(state_dim, action_dim).cuda()
        self._critic_2_target = CriticNetwork(state_dim, action_dim ).cuda()
        self._critic_2_target.load_state_dict(self._critic_2.state_dict())
        self._critic_2_optimizer = optim.Adam(self._critic_2.parameters(), lr = learning_rate)
        
        self._loss_function = nn.MSELoss()
        #Summary Writer
        if not os.path.exists("./results"):
            os.makedirs("./results")
        self.model_name = "{}_{}".format(model_name, datetime.now().strftime('%d-%m_%I-%M'))
        self.writer_name = "./results/{}".format(self.model_name)
        self.eval_writer_name = "./results/%s_eval"%self.model_name
        self.trained_path = "./trained_models/{}".format(self.model_name)
    
    def learn(self, total_timesteps = 10000, log_interval=100 , max_episode_length = 1e6):
        stats = EpisodeStats(episode_lengths = [], episode_rewards = [])
        eval_writer = SummaryWriter(self.eval_writer_name)
        writer = SummaryWriter(self.writer_name)
        episode = 0
        s = self.env.reset()
        episode_reward, episode_length, avg_reward = 0,0,0 #avg_reward over log_interval timesteps
        best_reward = -np.inf

        for t in range(1, total_timesteps+1):
            a = self._actor.predict(tt(s), self.env, deterministic = False)#sample action and scale it to action space
            a = a.cpu().detach().numpy()
            ns, r, done, _ = self.env.step(a)
            self._replay_buffer.add_transition(s, a, r, ns, done)
            s = ns
            avg_reward+=r
            episode_reward +=r
            episode_length +=1           
            
            #Replay buffer has enough data
            if(self._replay_buffer.__len__()>= self.batch_size and not done):
                sample = self._replay_buffer.sample(self.batch_size)
                batch_states, batch_actions, batch_rewards, batch_next_states, batch_terminal_flags = sample

                if(t % self.train_freq == 0): #If we have done train_freq env. steps
                    for n in range(self.gradient_steps):
                        #---------------- Critic networks update -------------#
                        with torch.no_grad():
                            next_actions, log_probs = self._actor.sample(batch_next_states, reparameterize = False)
                            next_actions *= self.env.action_space.high[0] #scale action between high and -high action space

                            target_critic_value = torch.min(
                                self._critic_1_target(batch_next_states, next_actions),
                                self._critic_2_target(batch_next_states, next_actions) )
                            
                            td_target = batch_rewards + (1 - batch_terminal_flags) * self._gamma * \
                                    (target_critic_value - self.ent_coef * log_probs)     
                        #Critic 1
                        curr_prediction_c1 = self._critic_1(batch_states, batch_actions)                
                        self._critic_1_optimizer.zero_grad()
                        loss_c1 = self._loss_function(curr_prediction_c1, td_target.detach())
                        loss_c1.backward()
                        #nn.utils.clip_grad_norm_(self._critic_1.parameters(),1)
                        self._critic_1_optimizer.step()
                        
                        #Critic 2
                        curr_prediction_c2 = self._critic_2(batch_states, batch_actions)
                        self._critic_2_optimizer.zero_grad()
                        loss_c2 = self._loss_function(curr_prediction_c2, td_target.detach())
                        loss_c2.backward()
                        #nn.utils.clip_grad_norm_(self._critic_2.parameters(),1)
                        self._critic_2_optimizer.step()

                        #---------------- Policy network update -------------#
                        predicted_actions, log_probs = self._actor.sample(batch_states, reparameterize = True)
                        critic_value = torch.min(
                            self._critic_1(batch_states, predicted_actions),
                            self._critic_2(batch_states, predicted_actions) )
                        #Actor update/ gradient ascent
                        self._actor_optimizer.zero_grad()
                        policy_loss = (self.ent_coef * log_probs - critic_value).mean()
                        policy_loss.backward()
                        self._actor_optimizer.step()

                        #---------------- Entropy network update -------------#
                        self.ent_coef_optimizer.zero_grad()
                        ent_coef_loss = -(self.log_ent_coef * (log_probs + self.target_entropy).detach()).mean()
                        ent_coef_loss.backward()
                        self.ent_coef_optimizer.step()
                        self.ent_coef = self.log_ent_coef.exp()

                        #------------------ Target Networks update -------------------#
                        soft_update(self._critic_1_target, self._critic_1, self.tau)
                        soft_update(self._critic_2_target, self._critic_2, self.tau)

            if(done or episode_length >= max_episode_length): #End episode
                stats.episode_lengths.append(episode_length)
                stats.episode_rewards.append(episode_reward)
                print("Episode %d: %d Steps, Reward: %.3f, total timesteps: %d/%d "%(episode, episode_length, episode_reward, t, total_timesteps))
                #Summary Writer
                writer.add_scalar('train/episode_reward', episode_reward, episode)
                writer.add_scalar('train/episode_length', episode_length, episode)

                if(episode_reward>best_reward and episode>20):#more than 50 episodes..
                    if(episode_reward<0):
                        self.save(self.trained_path+"_r-neg%d.pth"%(np.abs(round(episode_reward))))
                    else:
                        self.save(self.trained_path+"_r-%d.pth"%(round(episode_reward)))
                    best_reward = episode_reward
                
                #Reset everything
                episode+=1
                episode_reward, episode_length = 0,0
                s = self.env.reset()

            if(t%log_interval==0):
                if(episode<=0):
                    avg_reward = np.sum(avg_reward)
                else:
                    avg_reward = avg_reward/episode #episode is a counter of how many episodes there are
                writer.add_scalar('train/average_episode_reward', avg_reward, t)
                if(self.eval_env is not None):
                    self.log_tensorboard(self.eval_env, eval_writer,step = t,\
                                        max_episode_length=max_episode_length, model_name=self.model_name) #timesteps in x axis
                avg_reward = 0

        if(self.eval_env is not None):
            print("End of training evaluation:")
            self.evaluate(self.eval_env, model_name = self.model_name, print_all_episodes = True, write_file = True)
        #self.save_stats(stats, self.writer_name)
        return stats

    def log_tensorboard(self, env, writer, step, max_episode_length, model_name="sac", write_file = False):
        mean_reward, mean_length = self.evaluate(env,max_episode_length=max_episode_length,\
                                                 model_name = model_name, write_file = write_file)
        writer.add_scalar('eval/mean_reward', mean_reward, step)
        writer.add_scalar('eval/mean_length', mean_length, step)

    def evaluate(self, env, max_episode_length = 150, n_episodes = 10, model_name = "sac",\
                 print_all_episodes = False, write_file = True, render = False):
        stats = EpisodeStats(episode_lengths = [], episode_rewards = [])
        for episode in range(n_episodes):
            s = env.reset()
            episode_length, episode_reward = 0,0
            done = False
            while( episode_length < max_episode_length and not done):
                a = self._actor.predict(tt(s), env, deterministic = True)#sample action and scale it to action space
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
        torch.save({
            'actor_dict': self._actor.state_dict(),
            #'actor_target_dict': self._actor_target.state_dict(),
            'actor_optimizer_dict': self._actor_optimizer.state_dict(),
            'critic_1_dict': self._critic_1.state_dict(),
            'critic_1_target_dict': self._critic_1_target.state_dict(),
            'critic_1_optimizer_dict': self._critic_1_optimizer.state_dict(),
            'critic_2_dict': self._critic_2.state_dict(),
            'critic_2_target_dict': self._critic_2_target.state_dict(),
            'critic_2_optimizer_dict': self._critic_2_optimizer.state_dict(),
            # 'value_dict': self._critic.state_dict(),
            # 'value_target_dict': self._critic_target.state_dict(),
            # 'value_optimizer_dict': self._critic_optimizer.state_dict()
            }, path)
  
    def load(self, path):
        if os.path.isfile(path):
            print("Loading checkpoint")
            checkpoint = torch.load(path)

            self._actor.load_state_dict(checkpoint['actor_dict'])
            #self._actor_target.load_state_dict(checkpoint['actor_target_dict'])
            self._actor_optimizer.load_state_dict(checkpoint['actor_optimizer_dict'])

            self._critic_1.load_state_dict(checkpoint['critic_1_dict'])
            self._critic_1_target.load_state_dict(checkpoint['critic_1_target_dict'])
            self._critic_1_optimizer.load_state_dict(checkpoint['critic_1_optimizer_dict'])

            self._critic_2.load_state_dict(checkpoint['critic_2_dict'])
            self._critic_2_target.load_state_dict(checkpoint['critic_2_target_dict'])
            self._critic_2_optimizer.load_state_dict(checkpoint['critic_2_optimizer_dict'])

            # self._value.load_state_dict(checkpoint['value_dict'])
            # self._value_target.load_state_dict(checkpoint['value_target_dict'])
            # self._value_optimizer.load_state_dict(checkpoint['value_optimizer_dict'])
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
