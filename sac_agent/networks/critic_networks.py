import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal
from sac_agent.networks.networks_common import *
from sac_agent.sac_utils.utils import tt, get_activation_fn
import numpy as np
#q function     
class CriticNetwork(nn.Module):
  def __init__(self, state_dim, action_dim, activation = "relu", hidden_dim=256 ):
    super(CriticNetwork, self).__init__()
    self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
    self.fc2 = nn.Linear(hidden_dim, hidden_dim)
    self.q = nn.Linear(hidden_dim, 1)
    self._activation = get_activation_fn(activation)

  def forward(self, states, actions):
    x = torch.cat((states,actions), -1)
    x = self._activation(self.fc1(x))
    x = self._activation(self.fc2(x))
    return self.q(x).squeeze()

#q function     
class CNNCritic(nn.Module):
  def __init__(self, obs_space, action_dim, hidden_dim=256, activation = "relu"):
    super(CNNCritic, self).__init__()
    _position_shape = get_pos_shape(obs_space)
    self.cnn_depth = get_depth_network(obs_space, out_feat = 8, activation = activation)
    self.cnn_img = get_img_network(obs_space, out_feat = 8, activation = activation)

    out_feat = 16 if self.cnn_depth is not None else 8
    out_feat += _position_shape + action_dim

    self._activation = activation
    self.fc1 = nn.Linear(out_feat, hidden_dim)
    self.q = nn.Linear(hidden_dim, 1)

  def forward(self, states, actions):
    features = get_concat_features(states, self.cnn_img, self.cnn_depth)

    x = torch.cat((features,actions), -1)
    x = F.elu(self.fc1(x))
    x = self.q(x).squeeze()
    return x