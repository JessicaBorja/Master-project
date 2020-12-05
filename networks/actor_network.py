import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal

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
