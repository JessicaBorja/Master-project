import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from collections import namedtuple
import os
import pickle
import importlib

EpisodeStats = namedtuple(
                "Stats",
                ["episode_lengths", "episode_rewards", "validation_reward"])


def get_nets(img_obs, obs_space):
    if(img_obs):
        # policy = "CNNPolicy"
        # critic = "CNNCritic"
        policy = "legacy_CNNPolicy"
        critic = "legacy_CNNCritic"
    else:
        obs_space = obs_space.shape[0]
        policy = "ActorNetwork"
        critic = "CriticNetwork"
    policy_net = getattr(
        importlib.import_module("sac_agent.networks.actor_network"),
        policy)
    critic_net = getattr(
        importlib.import_module("sac_agent.networks.critic_network"),
        critic)
    return policy_net, critic_net, obs_space


def tt(x):
    if isinstance(x, dict):
        dict_of_list = {}
        for key, val in x.items():
            dict_of_list[key] = Variable(torch.from_numpy(val).float().cuda(),
                                         requires_grad=False)
        return dict_of_list
    else:
        return Variable(torch.from_numpy(x).float().cuda(), 
                        requires_grad=False)

def soft_update(target, source, tau):
  for target_param, param in zip(target.parameters(), source.parameters()):
    target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
  soft_update(target, source, 1.0)

def get_activation_fn(non_linearity):
  if(non_linearity == "elu"):
    return F.elu
  elif (non_linearity == "leaky_relu"):
    return F.leaky_relu
  else:#relu
    return F.relu

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