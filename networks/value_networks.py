import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal

#q function     
class CriticNetwork(nn.Module):
  def __init__(self, state_dim, action_dim, non_linearity = F.relu, hidden_dim=256 ):
    super(CriticNetwork, self).__init__()
    self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
    self.fc2 = nn.Linear(hidden_dim, hidden_dim)
    self.q = nn.Linear(hidden_dim, 1)
    self._non_linearity = non_linearity

  def forward(self, states, actions):
    x = torch.cat((states,actions), -1)
    x = self._non_linearity(self.fc1(x))
    x = self._non_linearity(self.fc2(x))
    return self.q(x).squeeze()

#q function     
class CNNCritic(nn.Module):
  def __init__(self, state_dim, action_dim, non_linearity = F.relu, hidden_dim=256 ):
    super(CNNCritic, self).__init__()
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
    self._q = CriticNetwork(_position_shape + 16, action_dim,\
                             non_linearity= non_linearity, hidden_dim=hidden_dim)

  def forward(self, states, actions):
      img, pos = states['rgb_obs'], states['position']

      batch_size = img.shape[0]
      x = F.relu(self.conv1(img))
      x = F.relu(self.conv2(x))
      x = F.relu(self.conv3(x))
      x = F.relu(self.fc1(x.view(batch_size,-1))) #bs, 16
      
      b_states = torch.cat((pos,x), dim=-1)
      x = self._q(b_states, actions)
      return x

  def calc_out_size(self,w,h,kernel_size,padding,stride):
      width = (w - kernel_size +2*padding)//stride + 1
      height = (h - kernel_size +2*padding)//stride + 1
      return width, height

#value function     
class ValueNetwork(nn.Module):
  def __init__(self, state_dim, action_dim, non_linearity = F.relu, hidden_dim=256):
    super(ValueNetwork, self).__init__()
    self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
    self.fc2 = nn.Linear(hidden_dim, hidden_dim)
    self.v = nn.Linear(hidden_dim, hidden_dim)
    self._non_linearity = non_linearity

  def forward(self, states, actions):
    x = torch.cat((states,actions), -1)
    x = self._non_linearity(self.fc1(x))
    x = self._non_linearity(self.fc2(x))
    return self.v(x).squeeze()