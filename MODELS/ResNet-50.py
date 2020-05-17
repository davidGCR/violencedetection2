
import torch.nn as nn
from torchvision import models
from util import * 
from Identity import *
import torch
import constants
import MODELS.Pooling as Pooling

class ViolenceModelResNet(nn.Module):
    def __init__(self, num_classes, numDiPerVideos, model_name, joinType ,feature_extract, inference=False):
        super(ViolenceModelResNet, self).__init__()
        self.numDiPerVideos = numDiPerVideos
        self.inference = inference
        self.num_classes = num_classes
        self.joinType = joinType
        self.model_ft = None
        self.model_ft = models.resnet50(pretrained=True)
        self.num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = Identity()
        self.bn = nn.BatchNorm2d(2048)
        
        self.convLayers = nn.Sequential(*list(self.model_ft.children())[:-2])  # to tempooling
        model_ft = None
        self.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d((1,1))

        # set_parameter_requires_grad(self.model_ft, feature_extract)
        set_parameter_requires_grad(self.convLayers, feature_extract)
        self.linear = nn.Linear(2048, self.num_classes)
    

    def forward(self, x):
        if self.numDiPerVideos == 1:
            # print('ókokokokok', x.size())
            x = self.convLayers(x)  #torch.Size([8, 512, 7, 7])
            x = self.AdaptiveAvgPool2d(x) #  torch.Size([8, 512, 1, 1])
            # print('model_ft out: ',x.size())
            x = torch.flatten(x, 1) # torch.Size([8, 512])
            # print('1di x out: ',x.size()) 
        else: ##torch.Size([8, 1, 3, 224, 224])
            if self.joinType == constants.TEMP_MAX_POOL:
                x = Pooling.maxTemporalPool(x, self.convLayers)
                # print('max pooling: ',x.size())
                x = self.bn(x)
                x = self.AdaptiveAvgPool2d(x)
                # print('AdaptiveAvgPool2d out: ', x.size())
                x = torch.flatten(x, 1)
                x = torch.add(x,z)
                x = torch.add(x, y)
                x = self.bn(x)
                x = self.AdaptiveAvgPool2d(x)
                x = torch.flatten(x, 1)
        x = self.linear(x)
        return x

def maxTemporalPool(x, fn_encoder):
    numDiPerVideos = x.size()[0]
    lista = []
    for dimage in range(0, numDiPerVideos):
        feature = fn_encoder(x[dimage]) #torch.Size([8, 512, 7, 7])
        lista.append(feature)

    minibatch = torch.stack(lista, 0) #torch.Size([2, 8, 512, 7, 7])
    # print('minibatch out:', minibatch.size())
    minibatch = minibatch.permute(1, 0, 2, 3, 4) #torch.Size([8, 2, 512, 7, 7])
    # print('minibatch permuted out:', minibatch.size())
    
    tmppool = nn.MaxPool2d((numDiPerVideos, 1))
    lista_minibatch = []
    for idx in range(minibatch.size()[0]):
        out = tempMaxPooling(minibatch[idx], tmppool) #torch.Size([512, 7, 7])
        lista_minibatch.append(out)

    feature = torch.stack(lista_minibatch, 0) #torch.Size([bs, 512,7,7])
    # print('feature output: ', feature.size())
    return feature
  