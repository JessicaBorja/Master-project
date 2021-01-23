import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal
from utils.utils import tt
import cv2
import numpy as np 

class CNNCommon(nn.Module):
  def __init__(self, in_channels, input_size, out_feat): 
    super(CNNCommon, self).__init__()
    w,h = self.calc_out_size(input_size,input_size,8,0,4)
    w,h = self.calc_out_size(w,h,4,0,2)
    w,h = self.calc_out_size(w,h,3,0,1)

    self.conv1 = nn.Conv2d(in_channels, 32, 8, stride=4)
    self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
    self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
    self.fc1 = nn.Linear(w*h*64, out_feat)


  def forward(self, x):
    # cv2.imshow("forward pass", np.uint8(np.expand_dims(img[0].cpu().numpy(),-1)) )
    # cv2.waitKey(1)
    if(len(x.shape) == 3):
      x = x.unsqueeze(0)
    batch_size = x.shape[0]
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    x = F.relu(self.fc1(x.view(batch_size,-1))).squeeze() #bs, out_feat
    return x

  def calc_out_size(self,w,h,kernel_size,padding,stride):
    width = (w - kernel_size +2*padding)//stride + 1
    height = (h - kernel_size +2*padding)//stride + 1
    return width, height