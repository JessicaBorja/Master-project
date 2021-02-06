import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal
from sac_agent.sac_utils.utils import tt, get_activation_fn
import cv2
import numpy as np 
from sac_agent.networks.networks_common import *

#policy
class ActorNetwork(nn.Module):
  def __init__(self, state_dim, action_dim, action_max, activation = "relu", hidden_dim=256):
    super(ActorNetwork, self).__init__()
    self.fc1 = nn.Linear(state_dim, hidden_dim)
    self.fc2 = nn.Linear(hidden_dim, hidden_dim)
    self.mu = nn.Linear(hidden_dim, action_dim)
    self.sigma = nn.Linear(hidden_dim, action_dim)
    self._activation = get_activation_fn(activation)
    self.action_max = action_max

  def forward(self, x):
    x = self._activation(self.fc1(x))
    x = self._activation(self.fc2(x))
    mu =  self.mu(x)
    sigma = F.softplus(self.sigma(x))
    #ensure sigma is between "0" and 1
    return mu, sigma

  #return action scaled to env
  def act(self, curr_obs, deterministic = False, reparametrize = False):
    mu, sigma = self.forward(curr_obs)#.squeeze(0)
    log_probs = None
    if(deterministic):
      action = torch.tanh(mu) * self.action_max
    else:
      dist = Normal(mu, sigma)
      if(reparametrize):
          sample = dist.rsample()
      else:
          sample = dist.sample()
      action = torch.tanh(sample) * self.action_max
      #For updating policy, Apendix of SAC paper
      #unsqueeze because log_probs is of dim (batch_size, action_dim) but the torch.log... is (batch_size)
      log_probs = dist.log_prob(sample) - torch.log((1 - torch.tanh(sample).pow(2) + 1e-6)) #+1e-6 to avoid no log(0)
      log_probs = log_probs.sum(-1)#, keepdim=True)

    return action, log_probs

class CNNPolicy(nn.Module):
  def __init__(self, obs_space, action_dim, action_max, activation = "relu", hidden_dim=256): 
    super(CNNPolicy, self).__init__()
    self.action_max = action_max
    _position_shape = get_pos_shape(obs_space)
    self.cnn_depth = get_depth_network(obs_space, out_feat = 8, activation = activation)
    self.cnn_img = get_img_network(obs_space, out_feat = 8, activation = activation)

    out_feat = 16 if self.cnn_depth is not None else 8
    out_feat += _position_shape

    self.fc1 = nn.Linear(out_feat, hidden_dim)
    self.mu = nn.Linear(hidden_dim, action_dim)
    self.sigma = nn.Linear(hidden_dim, action_dim)

  def forward(self, obs):
    features = get_concat_features(obs, self.cnn_img, self.cnn_depth)

    x = F.elu(self.fc1(features))
    mu =  self.mu(x)
    sigma = F.softplus(self.sigma(x))
    return mu, sigma

  #return action scaled to env
  def act(self, curr_obs, deterministic = False, reparametrize = False):
    mu, sigma = self.forward(curr_obs)#.squeeze(0)
    log_probs = None
    if(deterministic):
      action = torch.tanh(mu) * self.action_max
    else:
      dist = Normal(mu, sigma)
      if(reparametrize):
          sample = dist.rsample()
      else:
          sample = dist.sample()
      action = torch.tanh(sample) * self.action_max
      #For updating policy, Apendix of SAC paper
      #unsqueeze because log_probs is of dim (batch_size, action_dim) but the torch.log... is (batch_size)
      log_probs = dist.log_prob(sample) - torch.log((1 - torch.tanh(sample).pow(2) + 1e-6)) #+1e-6 to avoid no log(0)
      log_probs = log_probs.sum(-1)#, keepdim=True)
    return action, log_probs

