import torch.nn as nn
from torchvision import models
# from util import *
from UTIL.parameters import set_parameter_requires_grad
# from tempPooling import *
import MODELS.Pooling as Pooling
import MODELS.Identity as Identity
import torch
import constants


class AlexNet(nn.Module): ##ViolenceModel2
  def __init__(self, num_classes, numDiPerVideos, joinType, feature_extract):
      super(AlexNet, self).__init__()
      self.numDiPerVideos = numDiPerVideos
      self.joinType = joinType
      self.num_classes = num_classes
      self.model = models.alexnet(pretrained=True)
      set_parameter_requires_grad(self.model, feature_extract)
      self.linear = None
      self.model = nn.Sequential(*list(self.model.children())[:-2]) # to tempooling
      # self.linear = nn.Linear(4096,2)
      self.linear = nn.Linear(256 * 6 * 6, self.num_classes)
      self.tmpPooling = nn.MaxPool2d((numDiPerVideos, 1))
      # num_ftrs = model_ft.classifier[6].in_features
      # self.model.classifier[6] = nn.Linear(num_ftrs,num_classes)
      
  
  def forward(self, x):
    # numDynamicImages = x.size(0)
    # lista = []
    x = x.permute(1, 0, 2, 3, 4)
    # print('x_in: ', x.size())
    batch_size, timesteps, C, H, W = x.size()
    c_in = x.view(batch_size * timesteps, C, H, W)
    # print('cin: ', c_in.size())
    c_out = self.model(c_in)
    # print('cout: ', c_out.size())
    x = torch.flatten(c_out, 1)
    # print('flatten: ', x.size())
    # Re-structure the data and then temporal max-pool.
    x = x.view(batch_size, timesteps, 256 * 6 * 6)
    # print('Re-structure: ', x.size())
    x = x.max(dim=1).values
    # print('maxpooling: ', x.size())
    x = self.linear(x)

    # print('Finalx: ', x.size())
    return x
  

