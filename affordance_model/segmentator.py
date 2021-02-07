import numpy as np
import torch
import torchvision as tv
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal
import itertools

#https://github.com/qubvel/segmentation_models.pytorch
class Segmentator:
  #Replay buffer for experience replay. Stores transitions.
  def __init__(self, max_size, dict_state = False):
      return