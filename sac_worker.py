from datetime import datetime
import os, sys
import json, pickle
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal
import time
from utils.replay_buffer import ReplayBuffer
from utils.utils import EpisodeStats, tt, soft_update
from utils.networks import ActorNetwork, CriticNetwork

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from hpbandster.core.worker import Worker
import logging
from sac import SAC

class SACWorker(Worker):
    def __init__(self, hyperparameters, eval_config):
        self.fixed_hyperparams = hyperparameters
        self.eval_config = eval_config

    def compute(self, config, budget, working_directory, *args, **kwargs):
        """
        Simple example for a compute function using a feed forward network.
        It is trained on the MNIST dataset.
        The input parameter "config" (dictionary) contains the sampled configurations passed by the bohb optimizer
        """
        actor_lr = config["actor_lr"]
        critic_lr = config["critic_lr"]
        alpha_lr = config["alpha_lr"]
        model = SAC(actor_lr=actor_lr, critic_lr=critic_lr, alpha_lr=alpha_lr, **self.fixed_hyperparams)
        stats = self.model.learn(total_timesteps = budget, log_interval = 1000)
        train_reward = np.max(stats.episode_rewards)
        eval_mean_reward, _ = model.evaluate(**self.eval_config)

        return ({ #want to maximize reward then minimize negative reward
                'loss': - eval_mean_reward, # remember: HpBandSter always minimizes!
                'info': { 'train accuracy': train_reward,
                            'validation accuracy': eval_mean_reward,
                        } })

    @staticmethod
    def get_configspace():
        """
        It builds the configuration space with the needed hyperparameters.
        It is easily possible to implement different types of hyperparameters.
        Beside float-hyperparameters on a log scale, it is also able to handle categorical input parameter.
        :return: ConfigurationsSpace-Object
        """
        cs = CS.ConfigurationSpace()
        actor_lr = CSH.UniformFloatHyperparameter('actor_lr', lower=1e-6, upper=1e-1, default_value='1e-2', log=True)
        critic_lr = CSH.UniformFloatHyperparameter('critic_lr', lower=1e-6, upper=1e-1, default_value='1e-2', log=True)
        alpha_lr = CSH.UniformFloatHyperparameter('alpha_lr', lower=1e-6, upper=1e-1, default_value='1e-2', log=True)

        cs.add_hyperparameters([actor_lr, critic_lr, alpha_lr])
        return cs

