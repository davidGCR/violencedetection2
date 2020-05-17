import torch.nn as nn
from torchvision import models
# from util import *
from UTIL.parameters import set_parameter_requires_grad
# from tempPooling import *
import MODELS.Pooling as Pooling
import MODELS.Identity as Identity
import torch
import constants


class ViolenceModelAlexNet(nn.Module): ##ViolenceModel2
  def __init__(self, num_classes, numDiPerVideos, joinType, feature_extract):
      super(ViolenceModelAlexNet, self).__init__()
      self.numDiPerVideos = numDiPerVideos
      self.joinType = joinType
      self.num_classes = num_classes
      self.model = models.alexnet(pretrained=True)
      set_parameter_requires_grad(self.model, feature_extract)
      
      # if self.joinType == 'cat':
      #   self.num_ftrs = self.model.classifier[6].in_features
      #   self.model.classifier = self.model.classifier[:-1]
      #   self.linear = nn.Linear(self.num_ftrs*numDiPerVideos,self.num_classes)
      self.linear = None
      if self.joinType == constants.TEMP_MAX_POOL:
        self.model = nn.Sequential(*list(self.model.children())[:-2]) # to tempooling
        # self.linear = nn.Linear(4096,2)
        self.linear = nn.Linear(256*6*6,self.num_classes)
      
  
  def forward(self, x):
    # shape = x.size()
    # if self.numDiPerVideos == 1:
    #   x = self.model(x)
    #   x = torch.flatten(x, 1)
    #   # print('x: ',x.size())
    # else:
    #   if self.joinType == 'cat':
    #     x = self.getFeatureVectorCat(x)
    #   elif self.joinType == constants.TEMP_MAX_POOL:
    #     x = self.getFeatureVectorTempPool(x)
    
    # x = self.linear(x)
    return x
  
  # def getFeatureVectorCat(self, x):
  #   lista = []
  #   for dimage in range(0, self.numDiPerVideos):
  #     feature = self.model(x[dimage])
  #     lista.append(feature)
  #   x = torch.cat(lista, dim=1)  
  #   return x

  # def getFeatureVectorTempPool(self, x):
  #   lista = []
  #   seqLen = self.numDiPerVideos
  #   for dimage in range(0, seqLen):
  #     feature = self.model(x[dimage])
  #     lista.append(feature)
  #   minibatch = torch.stack(lista, 0)
  #   minibatch = minibatch.permute(1, 0, 2, 3, 4)
  #   num_dynamic_images = self.numDiPerVideos
  #   tmppool = nn.MaxPool2d((num_dynamic_images, 1))
  #   lista_minibatch = []
  #   for idx in range(minibatch.size()[0]):
  #       out = tempMaxPooling(minibatch[idx], tmppool)
  #       lista_minibatch.append(out)
  #   feature = torch.stack(lista_minibatch, 0)
  #   feature = torch.flatten(feature, 1)
  #   # feature = self.classifier(feature)
  #   return feature



  #   class ViolenceModelAlexNetV1(nn.Module): ##ViolenceModel
  # def __init__(self, seqLen, joinType,feature_extract):
  #     super(ViolenceModelAlexNetV1, self).__init__()
  #     self.seqLen = seqLen
  #     self.joinType = joinType
  #     self.alexnet = models.alexnet(pretrained=True)
  #     self.feature_extract = feature_extract
  #     set_parameter_requires_grad(self.alexnet, feature_extract)
      
  #     self.model = nn.Sequential(*list(self.alexnet.features.children()))

  #     if self.joinType == 'cat':
        
  #       self.linear = nn.Linear(256 * 6 * 6*seqLen,2)
  #     elif self.joinType == 'tempMaxPool':
  #       self.linear = nn.Linear(256 * 6 * 6,2)
  #     self.alexnet = None

  #     # self.linear = nn.Linear(256*6*6*seqLen,2)
  #     # self.alexnet = None

  # def forward(self, x):
  #   if self.joinType == 'cat':
  #     x = self.getFeatureVectorCat(x)  
  #   elif self.joinType == 'tempMaxPool':
  #     x = self.getFeatureVectorTempPool(x)  

  #   # lista = []
  #   # for dimage in range(0, self.seqLen):
  #   #   feature = self.convNet(x[dimage])
  #   #   feature = feature.view(feature.size(0), 256 * 6 * 6)
  #   #   lista.append(feature)
  #   # x = torch.cat(lista, dim=1)  
  #   x = self.linear(x)
  #   return x
  
  # def getFeatureVectorTempPool(self, x):
  #   lista = []
  #   for dimage in range(0, self.seqLen):
  #     feature = self.convNet(x[dimage])
  #     lista.append(feature)

  #   minibatch = torch.stack(lista, 0)
  #   minibatch = minibatch.permute(1, 0, 2, 3, 4)
  #   num_dynamic_images = self.seqLen
  #   tmppool = nn.MaxPool2d((num_dynamic_images, 1))
  #   lista_minibatch = []
  #   for idx in range(minibatch.size()[0]):
  #       out = tempMaxPooling(minibatch[idx], tmppool)
  #       lista_minibatch.append(out)

  #   feature = torch.stack(lista_minibatch, 0)
  #   feature = torch.flatten(feature, 1)
  #   return feature
  
  # def getFeatureVectorCat(self, x):
  #   lista = []
  #   for dimage in range(0, self.seqLen):
  #     feature = self.convNet(x[dimage])
  #     feature = feature.view(feature.size(0), 256 * 6 * 6)
  #     lista.append(feature)
  #   x = torch.cat(lista, dim=1) 
  #   return x
############################################################################
############################################################################
############################################################################
