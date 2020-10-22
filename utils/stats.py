import matplotlib.pyplot as plt
import matplotlib.colors as Color
import pandas as pd
import numpy as np
import cloudpickle 
from pathlib import Path
import os
#Saves a file of a list of staths in a given path
def save_stats(lst, path):
  with open(path, "wb") as fp:   #Pickling
    cloudpickle.dump(lst, fp)

#loads a list of stats from a given path (full path or relative path)
def load_stats(path):
  if os.path.isfile(path):
    with open(path, "rb") as fp:   # Unpickling
      stats_lst = cloudpickle.load(fp)
      return stats_lst
  return None

#gets a list of stats and returns the mean and std (which are again lists of the lenght of the episodes)
def get_mean_and_std(stats_lst, custom = False):
  #mean and std of stats
  episodes_stats =  np.array([stat for name, stat in stats_lst])
  if(custom):
    rewards = episodes_stats[:,2]
  else:
    rewards = episodes_stats[:,1]
  lengths = episodes_stats[:,0]
  
  mean_length = np.mean(lengths, axis=0)
  std_length = np.std(lengths, axis=0)
  mean_rewards = np.mean(rewards, axis=0)
  std_rewards = np.std(rewards, axis=0)
  
  return mean_length, std_length, mean_rewards, std_rewards

def plot_multiple_means_join(mean_list, smoothing_window=10, plot_std = False, noshow=False, file_name = "x", title = "Mean rewards"):
  # Plot the episode length over time
  fig1,ax = plt.subplots(figsize=(7,5),sharey=True,sharex=True)  
  colormap = plt.get_cmap("plasma")
  num_models= len(mean_list)
  ax.set_prop_cycle(color =  [colormap(i) for i in np.linspace(0, 0.9, num_models)])

  for mean_tuple in mean_list:
    name, mean_r, std_r = mean_tuple
    mean_smoothed = pd.Series(mean_r).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(mean_smoothed,label = name)
  
  #plot
  plt.legend(loc='upper left')
  plt.xlabel('Episode')
  plt.ylabel("Episode Reward (Smoothed)")
  plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))  
  plt.suptitle(title,y=1.05, fontsize=18)
  plt.plot()
  
  fig1.savefig("./files/plots/" + file_name+'.png')
  if noshow:
      plt.close(fig1)
  else:
      plt.show(fig1)
      
def plot_multiple_means(mean_list, smoothing_window=10, plot_std = False, noshow=False, file_name = "x", title = "Mean rewards"):
  # Plot the episode length over time
  fig1,axs = plt.subplots(1,len(mean_list),figsize=(20,5),sharey=True,sharex=True)  
  colormap = plt.get_cmap("winter")
  num_models= len(mean_list)
  color_map = [colormap(i) for i in np.linspace(0, 0.9, num_models)]
  #ax.set_prop_cycle(color =  [colormap(i) for i in np.linspace(0, 0.9, num_models)])
  i=0
  ax0=axs[0]
  ax0.set_ylabel('Episode Reward (Smoothed)')
    
  for mean_tuple,ax in zip(mean_list,axs):
    name, mean_r, std_r = mean_tuple
    mean_smoothed = pd.Series(mean_r).rolling(smoothing_window, min_periods=smoothing_window).mean()
    std_smoothed = pd.Series(std_r).rolling(smoothing_window, min_periods=smoothing_window).mean()
    c= Color.to_hex(color_map[i][:-1])
    if(plot_std):
        x = np.arange(0,len(mean_r),1)
        ax.fill_between(x, mean_smoothed + std_smoothed, mean_smoothed - std_smoothed,color=c, alpha='0.2')
    ax.plot(mean_smoothed,color= c,label = name)
    ax.legend(loc='upper left')
    ax.set_xlabel('Episode')
    ax.set_title(name)
    i+=1
  
  #plot
  plt.plot()
  plt.suptitle(title,y=1.05, fontsize=18)
  plt.tight_layout()
  
  fig1.savefig("./files/plots/" + file_name+'.png')
  if noshow:
      plt.close(fig1)
  else:
      plt.show(fig1)
      
#given a list of stats plots the stats and the mean and std
#stats_lst is a list of tuples (name, stats)
def plot_multiple_stats(stats_lst, smoothing_window=10,show_all=True, custom=False, noshow=False, file_name = "x", title="x"):
  
  #mean and std of stats
  mean_l,std_l,mean_r,std_r = get_mean_and_std(stats_lst, custom)
  num_episodes = len(mean_l)
  x = np.arange(0,num_episodes,1)

  # Plot the episode length over time
  fig1,ax = plt.subplots(figsize=(10,5))  

  if(show_all):
    colormap = plt.get_cmap("winter")
    num_models= len(stats_lst)
    ax.set_prop_cycle(color = [colormap(i) for i in np.linspace(0, 0.9, num_models)])
    for stat_tuple in stats_lst:
      name, stats = stat_tuple
      plt.plot(stats.episode_lengths, label =  name)
  
  #plot mean
  plt.plot(mean_l, label ="Mean", color = 'black', linestyle = '--')
  plt.fill_between(x, mean_l+std_l, mean_l-std_l, color='purple', alpha='0.2')
  
  #plot
  plt.legend()
  plt.xlabel("Episode")
  plt.ylabel("Episode Length")
  plt.suptitle(title, y=1.05, fontsize=18)
  plt.title("Episode Length over Time")

  fig1.savefig("./files/plots/" + file_name + '_multiple_runs_episode_lengths.png')
  if noshow:
      plt.close(fig1)
  else:
      plt.show(fig1)

  # Plot the episode reward over time
  fig2,ax2 = plt.subplots(figsize=(10,5))
  if(show_all):
    ax2.set_prop_cycle(color = [colormap(i) for i in np.linspace(0, 0.9, num_models)])
    for stat_tuple in stats_lst:
      name, stats = stat_tuple
      if(custom):
        rewards_smoothed = pd.Series(stats.custom_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
      else: 
        rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
      plt.plot(rewards_smoothed,label = name)
  
  #plot mean
  mean_smoothed = pd.Series(mean_r).rolling(smoothing_window, min_periods=smoothing_window).mean()
  std_smoothed = pd.Series(std_r).rolling(smoothing_window, min_periods=smoothing_window).mean()
  plt.plot(mean_smoothed, label ="Mean", color = 'black',  linestyle = '--')
  plt.fill_between(x, mean_smoothed + std_smoothed, mean_smoothed - std_smoothed, color='purple', alpha='0.2')
  
  #plot
  plt.legend()
  plt.xlabel("Episode")
  plt.ylabel("Episode Reward (Smoothed)")
  plt.suptitle(title, y=1.05, fontsize=18)
  plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window), fontsize=10)

  fig2.savefig("./files/plots/" + file_name + '_multiple_runs_episode_rewards.png')
  if noshow:
      plt.close(fig2)
  else:
    plt.show(fig2)

#previous code from exercises to plot for 1 run (:
def plot_episode_stats(stats, smoothing_window=10, noshow=False, file_name = "x", title="x"):
  # Plot the episode length over time
  fig1 = plt.figure(figsize=(10,5))
  plt.plot(stats.episode_lengths)
  plt.xlabel("Episode")
  plt.ylabel("Episode Length")
  plt.title("Episode Length over Time")
  plt.suptitle(title)

  fig1.savefig("./files/plots/" + file_name +  '_episode_lengths.png')
  if noshow:
      plt.close(fig1)
  else:
      plt.show(fig1)

  # Plot the episode reward over time
  fig2 = plt.figure(figsize=(10,5))
  rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
  plt.plot(rewards_smoothed)
  plt.xlabel("Episode")
  plt.ylabel("Episode Reward (Smoothed)")
  plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
  plt.suptitle(title)

  fig2.savefig("./files/plots/" + file_name + '_episode_rewards.png')
  if noshow:
      plt.close(fig2)
  else:
      plt.show(fig2)