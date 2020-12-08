import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from collections import namedtuple
import os, pickle

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

def read_results(file_name, folder_name = "."):
    with open(os.path.join("%s/optimization_results/"%folder_name, file_name), 'rb') as fh:
        res = pickle.load(fh)

    id2config = res.get_id2config_mapping()
    incumbent = res.get_incumbent_id()
    print(id2config[incumbent])

if __name__ == "__main__":
    read_results(file_name = "optim_hinge_rn1_rs1.pkl", folder_name ="../outputs/hinge/2020-12-04/12-44-23/")