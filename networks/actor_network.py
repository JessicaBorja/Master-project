import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal
from utils.utils import tt
import cv2
import numpy as np 

#policy
class ActorNetwork(nn.Module):
  def __init__(self, state_dim, action_dim, action_max, non_linearity= F.relu, hidden_dim=256):
    super(ActorNetwork, self).__init__()
    self.fc1 = nn.Linear(state_dim, hidden_dim)
    self.fc2 = nn.Linear(hidden_dim, hidden_dim)
    self.mu = nn.Linear(hidden_dim, action_dim)
    self.sigma = nn.Linear(hidden_dim, action_dim)
    self._non_linearity = non_linearity
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
  def __init__(self, state_dim, action_dim, action_max,\
                non_linearity= F.relu, hidden_dim=256): 
    super(CNNPolicy, self).__init__()
    self.action_max = action_max
    self.non_linearity = F.relu

    _history_length = state_dim['rgb_obs'].shape[0]
    self.conv1 = nn.Conv2d(_history_length, 16, 5)
    self.conv2 = nn.Conv2d(16, 32, 3)
    self.conv3 = nn.Conv2d(32, 16, 5)

    _img_size = state_dim['rgb_obs'].shape[-1]
    w,h = self.calc_out_size(_img_size,_img_size,5,0,1)
    w,h = self.calc_out_size(w,h,3,0,1)
    w,h = self.calc_out_size(w,h,5,0,1)

    self.fc1 = nn.Linear(w*h*16, 16)
    _position_shape = state_dim['position'].shape[0]
    _position_shape = 0 ###
    self.policy_network = ActorNetwork(_position_shape + 16, action_dim, action_max,\
                                        non_linearity= non_linearity, hidden_dim=hidden_dim)

  def forward(self, x):
    img, pos = x['rgb_obs'], x['position']
    # cv2.imshow("forward pass", np.uint8(np.expand_dims(img[0].cpu().numpy(),-1)) )
    # cv2.waitKey(1)
    if(len(img.shape) == 3):
      img = img.unsqueeze(0)
    batch_size = img.shape[0]
    x = F.relu(self.conv1(img))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    x = F.relu(self.fc1(x.view(batch_size,-1))).squeeze() #bs, 16
    #x = self.policy_network(torch.cat((pos, x), dim=-1))
    x = self.policy_network(x) ###
    return x

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

  def calc_out_size(self,w,h,kernel_size,padding,stride):
    width = (w - kernel_size +2*padding)//stride + 1
    height = (h - kernel_size +2*padding)//stride + 1
    return width, height