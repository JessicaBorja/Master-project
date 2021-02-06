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

#cnn common takes the function directly, not the str
class CNNCommon(nn.Module):
  def __init__(self, in_channels, input_size, out_feat, non_linearity=F.relu): 
    super(CNNCommon, self).__init__()
    w,h = self.calc_out_size(input_size,input_size,8,0,4)
    w,h = self.calc_out_size(w,h,4,0,2)
    w,h = self.calc_out_size(w,h,3,0,1)

    self.conv1 = nn.Conv2d(in_channels, 32, 8, stride=4)
    self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
    self.conv3 = nn.Conv2d(64, 32, 3, stride=1)
    self.fc1 = nn.Linear(w*h*32, out_feat)
    self._non_linearity = non_linearity
      # nn.Conv2d(kwargs['in_channels'], 32, kernel_size=8, stride=4),
      # nn.ReLU(True),
      # nn.Conv2d(32, 64, kernel_size=4, stride=2),
      # nn.ReLU(True),
      # nn.Conv2d(64, 32, kernel_size=3, stride=1),
      # Flatten(),
      # nn.Linear(32*6*6, hidden_size),
      # nn.ReLU(True),

  def forward(self, x):
    # cv2.imshow("forward pass", np.uint8(np.expand_dims(img[0].cpu().numpy(),-1)) )
    # cv2.waitKey(1)
    if(len(x.shape) == 3):
      x = x.unsqueeze(0)
    batch_size = x.shape[0]
    x = self._non_linearity(self.conv1(x))
    x = self._non_linearity(self.conv2(x))
    x = self._non_linearity(self.conv3(x))
    x = self._non_linearity(self.fc1(x.view(batch_size,-1))).squeeze() #bs, out_feat
    return x

  def calc_out_size(self,w,h,kernel_size,padding,stride):
    width = (w - kernel_size +2*padding)//stride + 1
    height = (h - kernel_size +2*padding)//stride + 1
    return width, height