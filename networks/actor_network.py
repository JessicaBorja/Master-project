import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal
from utils.utils import tt, get_activation_fn
import cv2
import numpy as np 
from networks.common_archs import CNNCommon

#policy
class ActorNetwork(nn.Module):
  def __init__(self, state_dim, action_dim, action_max, activation = "relu", hidden_dim=256):
    super(ActorNetwork, self).__init__()
    self.fc1 = nn.Linear(state_dim, hidden_dim)
    self.fc2 = nn.Linear(hidden_dim, hidden_dim)
    self.mu = nn.Linear(hidden_dim, action_dim)
    self.sigma = nn.Linear(hidden_dim, action_dim)
    self._non_linearity = get_activation_fn(activation)
    self.action_max = action_max

  def forward(self, x):
    x = self._non_linearity(self.fc1(x))
    x = self._non_linearity(self.fc2(x))
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
  def __init__(self, state_dim, action_dim, action_max, activation = "relu",\
                use_img=True, use_pos=False, use_depth=False, hidden_dim=256): 
    super(CNNPolicy, self).__init__()
    self.action_max = action_max
    self._use_pos = use_pos
    self._use_depth = use_depth
    self._non_linearity = get_activation_fn(activation)
    #Image obs net
    _history_length = state_dim['rgb_obs'].shape[0]
    _img_size = state_dim['rgb_obs'].shape[-1]

    if(use_pos):
      _position_shape = state_dim['position'].shape[0]
    else:
      _position_shape = 0
    
    if(use_depth):
      self.cnn_depth = CNNCommon(1, _img_size, out_feat = 8, non_linearity = self._non_linearity)
      self.cnn_img = CNNCommon( _history_length, _img_size, out_feat = 8, non_linearity = self._non_linearity)
    else:
      self.cnn_img = CNNCommon( _history_length, _img_size, out_feat = 16, non_linearity = self._non_linearity)

    self.fc1 = nn.Linear(_position_shape + 16, hidden_dim)
    self.mu = nn.Linear(hidden_dim, action_dim)
    self.sigma = nn.Linear(hidden_dim, action_dim)

  def forward(self, x):
    img, depth, pos = x['rgb_obs'], x['depth_obs'], x['position']
    # cv2.imshow("forward pass", np.uint8(np.expand_dims(img[0].cpu().numpy(),-1)) )
    # cv2.waitKey(1)
    features = self.cnn_img(img)
    if(self._use_depth):
      depth_features = self.cnn_depth(depth)
      features = torch.cat((features, depth_features), dim=-1)
    if(self._use_pos):
      features = torch.cat((features, pos), dim=-1)

    # print("Img features", x)
    x = nn.ELU(self.fc1(features))
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

