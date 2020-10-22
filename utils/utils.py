import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from collections import namedtuple

EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])

def tt(ndarray):
  return Variable(torch.from_numpy(ndarray).float().cuda(), requires_grad=False)

def soft_update(target, source, tau):
  for target_param, param in zip(target.parameters(), source.parameters()):
    target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
  soft_update(target, source, 1.0)

def fan_in_uniform_init(tensor, fan_in=None):
    if fan_in is None:
        fan_in = tensor.size(-1)

    w = 1. / np.sqrt(fan_in)
    nn.init.uniform_(tensor, -w, w)