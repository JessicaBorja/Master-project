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

  def update(loss):
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

#value function     
class ValueNetwork(nn.Module):
  def __init__(self, state_dim, action_dim, non_linearity = F.relu, hidden_dim=256):
    super(ValueNetwork, self).__init__()
    self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
    self.fc2 = nn.Linear(hidden_dim, hidden_dim)
    self.v = nn.Linear(hidden_dim, hidden_dim)
    self._non_linearity = non_linearity
    # #self.layers.apply(self.init_weights)
    # init_w = 3e-3
    # fan_in_uniform_init(self.fc1.weight)
    # fan_in_uniform_init(self.fc2.weight)
    # fan_in_uniform_init(self.fc3.weight)
    # nn.init.uniform_(self.fc4.weight, -init_w, init_w)

  def forward(self, states, actions):
    x = torch.cat((states,actions), -1)
    x = self._non_linearity(self.fc1(x))
    x = self._non_linearity(self.fc2(x))
    return self.v(x).squeeze()

  def update(loss):
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()