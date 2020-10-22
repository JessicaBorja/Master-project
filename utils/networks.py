import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal

#policy
class ActorNetwork(nn.Module):
  def __init__(self, state_dim, action_dim, non_linearity= F.relu, hidden_dim=256):
    super(ActorNetwork, self).__init__()
    self.fc1 = nn.Linear(state_dim, hidden_dim)
    self.fc2 = nn.Linear(hidden_dim, hidden_dim)
    self.mu = nn.Linear(hidden_dim, action_dim)
    self.sigma = nn.Linear(hidden_dim, action_dim)
    self._non_linearity = non_linearity

  def forward(self, x):
    x = self._non_linearity(self.fc1(x))
    x = self._non_linearity(self.fc2(x))
    mu =  self.mu(x)
    sigma = F.softplus(self.sigma(x))
    #ensure sigma is between "0" and 1
    return mu, sigma

  #return action scaled to env
  def predict(self, curr_obs, env, deterministic = True):
      mu, sigma = self.forward(curr_obs)#.squeeze(0)
      if(deterministic):
        action = torch.tanh(mu) * env.action_space.high[0]
      else:
        dist = Normal(mu, sigma)
        sample = dist.sample()
        action = torch.tanh(sample) * env.action_space.high[0]

      return action
        
  def sample(self, curr_obs, reparameterize = True):
      mu, sigma = self.forward(curr_obs)#.squeeze(0)
      dist = Normal(mu, sigma)
      
      if(reparameterize):
          sample = dist.rsample()
      else:
          sample = dist.sample()
      
      action = torch.tanh(sample)
      #For updating policy, Apendix of SAC paper
      #unsqueeze because log_probs is of dim (batch_size, action_dim) but the torch.log... is (batch_size)
      log_probs = dist.log_prob(sample) - torch.log((1 - torch.tanh(sample).pow(2) + 1e-6)) #+1e-6 to avoid no log(0)
      log_probs = log_probs.sum(-1)#, keepdim=True)

      return action, log_probs

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