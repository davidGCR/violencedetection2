# import sys
# sys.path.insert(1, '/media/david/datos/PAPERS-SOURCE_CODE/MyCode')
import torch.nn as nn
from torchvision import models
from util import * 
from Identity import *
# from tempPooling import *
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
        if model_name == 'resnet18':
            self.model_ft = models.resnet18(pretrained=True)
        elif model_name == 'resnet34':
            self.model_ft = models.resnet34(pretrained=True)
        
        self.num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = Identity()
        # self.cat_linear = nn.Linear(512, )
        
        self.convLayers = nn.Sequential(*list(self.model_ft.children())[:-2])  # to tempooling
        model_ft = None
        self.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d((1,1))

        # set_parameter_requires_grad(self.model_ft, feature_extract)
        set_parameter_requires_grad(self.convLayers, feature_extract)
        if self.joinType == constants.JOIN_CONCATENATE:
            self.linear = nn.Linear(self.num_ftrs*self.numDiPerVideos,self.num_classes)
        elif self.joinType == constants.JOIN_TEMP_MAX_POOL:
            self.linear = nn.Linear(512, self.num_classes)
            # if model_name == 'resnet18' else nn.Linear(512*7*7,self.num_classes)
    
    def inferenceMode(self):
        self.inference = True

    def forward(self, x):
        # print('forward input size:',x.size())
        if self.inference:
            # batch_num = x.size()[0]
            # num_dyn_imgs_infer = x.size()[1]
            if self.numDiPerVideos > 1:
                x = torch.unsqueeze(x, dim=1)
                x = x.repeat(1, self.numDiPerVideos, 1, 1, 1)
                print('hereseeee: ', x.size())
                x = x.permute(1, 0, 2, 3, 4)  #[ndi, bs, c, h, w]
            

        if self.numDiPerVideos == 1:
            x = self.convLayers(x)  #torch.Size([8, 512, 7, 7])
            x = self.AdaptiveAvgPool2d(x) #  torch.Size([8, 512, 1, 1])
            # print('model_ft out: ',x.size())
            x = torch.flatten(x, 1) # torch.Size([8, 512])
            # print('1di x out: ',x.size()) 
        else: ##torch.Size([8, 1, 3, 224, 224])
            if self.joinType == constants.JOIN_CONCATENATE:
                # x = self.getFeatureVectorCat(x)
                x = Pooling.concatenate(x, self.numDiPerVideos, self.convLayers, self.AdaptiveAvgPool2d)
                # print('concatenate output size:',x.size())
                # x = self.AdaptiveAvgPool2d(x)
                # print('AdaptiveAvgPool2d output size:',x.size())
            elif self.joinType == constants.JOIN_TEMP_MAX_POOL:
                # print('x size: ', x.size())
                x = Pooling.maxTemporalPool(x, self.numDiPerVideos, self.convLayers)
                x = self.AdaptiveAvgPool2d(x)
                # print('AdaptiveAvgPool2d out: ', x.size())
                x = torch.flatten(x, 1)
        x = self.linear(x)
        # print('forward output: ', x.size())
        return x
  
    # def getFeatureVectorCat(self, x):
    #     lista = []
    #     for dimage in range(0, self.numDiPerVideos):
    #         feature = self.model_ft(x[dimage])
    #         # feature = torch.flatten(feature, 1)
    #         # feature = feature.view(feature.size(0), self.num_ftrs)
    #         lista.append(feature)
    #     x = torch.cat(lista, dim=1)
    #     return x