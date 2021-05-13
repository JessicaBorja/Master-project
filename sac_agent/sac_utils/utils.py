import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from collections import namedtuple
import os
import pickle
import importlib
from utils.img_utils import overlay_mask
import cv2

EpisodeStats = namedtuple(
                "Stats",
                ["episode_lengths", "episode_rewards", "validation_reward"])


# Both are numpy arrays
def show_mask_np(x, mask):
    # x.shape = [C, H, W]
    # mask.shape = [B, 2, H, W]
    show_mask = np.transpose(mask, (1, 2, 0))*255.0
    show_mask = cv2.normalize(show_mask, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)

    img = np.transpose(x, (1, 2, 0))*255.0
    img = cv2.normalize(img, None, 255, 0,
                        cv2.NORM_MINMAX, cv2.CV_8UC1)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    res = overlay_mask(show_mask, img, (0, 0, 255))
    cv2.imshow("paste", res)
    cv2.waitKey(1)


def set_init_pos(task, init_pos):
    if(task == "slide"):
        init_pos = [-1.1686195081948965, 1.5165126497924815, 1.7042540963745911, -1.6031852712241403, -2.5717679087567484, 2.331416872629473, -1.3006358472301627]
    elif(task == "drawer"):
        init_pos = [-0.4852725866746207, 1.0618989199760496, 1.3903811172536515, -1.7446581003391255, -1.1359501486104144, 1.8855365146855005, -1.3092771579652827]
    elif(task == "banana"):
        init_pos = [0.03740465833778156, 1.1844912206595481, 1.1330028132229706, -0.6702560563758552, -1.1188250499368455, 1.6153329476732947, -1.7078632665627795]
    elif(task == "hinge"):
        init_pos = [-0.3803066514807313, 0.931053115322005, 1.1668869976984892, -0.8602164833917604, -1.4818301463768684, 2.78299286093898, -1.7318962831826747]
    return init_pos


def get_nets(img_obs, obs_space, action_space):
    action_dim = action_space.shape[0]
    if(img_obs):
        print("SAC get_nets using: %s" % str([k for k in obs_space]))
        policy = "CNNPolicy"
        critic = "CNNCritic"
        # policy = "legacy_CNNPolicy"
        # critic = "legacy_CNNCritic"
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
    print("SAC get_nets: %s, \t%s" % (policy, critic))
    return policy_net, critic_net, obs_space, action_dim


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
        target_param.data.copy_(
            target_param.data * (1.0 - tau)
            + param.data * tau)


def hard_update(target, source):
    soft_update(target, source, 1.0)


def get_activation_fn(non_linearity):
    if(non_linearity == "elu"):
        return F.elu
    elif (non_linearity == "leaky_relu"):
        return F.leaky_relu
    else:  # relu
        return F.relu


def fan_in_uniform_init(tensor, fan_in=None):
    if fan_in is None:
        fan_in = tensor.size(-1)

    w = 1. / np.sqrt(fan_in)
    nn.init.uniform_(tensor, -w, w)


def read_results(file_name, folder_name="."):
    with open(
            os.path.join("%s/optimization_results/" % folder_name, file_name),
            'rb') as fh:
        res = pickle.load(fh)

    id2config = res.get_id2config_mapping()
    incumbent = res.get_incumbent_id()
    print(id2config[incumbent])
