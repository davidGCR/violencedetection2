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
        elif model_name == 'resnet50':
            self.model_ft = models.resnet50(pretrained=True)
        
        self.num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = Identity()
        if model_name == 'resnet18' or model_name == 'resnet34':
            self.bn = nn.BatchNorm2d(512)
        elif model_name == 'resnet50':
            self.bn = nn.BatchNorm2d(2048)
        
        self.convLayers = nn.Sequential(*list(self.model_ft.children())[:-2])  # to tempooling
        model_ft = None
        self.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d((1,1))

        # set_parameter_requires_grad(self.model_ft, feature_extract)
        set_parameter_requires_grad(self.convLayers, feature_extract)
        if self.joinType == constants.JOIN_CONCATENATE:
            self.linear = nn.Linear(self.num_ftrs*self.numDiPerVideos,self.num_classes)
        elif self.joinType == constants.TEMP_MAX_POOL or self.joinType == constants.MULT_TEMP_POOL or self.joinType == constants.TEMP_AVG_POOL or self.joinType == constants.TEMP_STD_POOL:
            if model_name == 'resnet18' or model_name == 'resnet34':
                self.linear = nn.Linear(512, self.num_classes)
            elif model_name == 'resnet50':
                self.linear = nn.Linear(2048, self.num_classes)
            # if model_name == 'resnet18' else nn.Linear(512*7*7,self.num_classes)
    
    def inferenceMode(self, numDiPerVideos):
        self.inference = True
        self.numDiPerVideos = numDiPerVideos
    
    def enableTransferLearning(self, feature_extract):
        set_parameter_requires_grad(self.convLayers, feature_extract)

    def forward(self, x):
        # x size= torch.Size([ndi, bs, 3, 224, 224])
        # print('forward input size:',x.size())
        if self.inference:
            # batch_num = x.size()[0]
            # num_dyn_imgs_infer = x.size()[1]
            if self.numDiPerVideos > 1:
                x = torch.unsqueeze(x, dim=1)
                # x = x.repeat(1, self.numDiPerVideos, 1, 1, 1) ###no se porque puse esto..
                # x = x.permute(1, 0, 2, 3, 4)  #[ndi, bs, c, h, w]
                print('x forward: ', x.size())
            

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
            elif self.joinType == constants.TEMP_MAX_POOL:
                x = Pooling.maxTemporalPool(x, self.convLayers)
                # print('max pooling: ',x.size())
                x = self.bn(x)
                x = self.AdaptiveAvgPool2d(x)
                # print('AdaptiveAvgPool2d out: ', x.size())
                x = torch.flatten(x, 1)
            elif self.joinType == constants.TEMP_AVG_POOL:
                x = Pooling.avgTemporalPool(x, self.numDiPerVideos, self.convLayers)
                # print('max pooling: ',x.size())
                x = self.bn(x)
                x = self.AdaptiveAvgPool2d(x)
                # print('AdaptiveAvgPool2d out: ', x.size())
                x = torch.flatten(x, 1)
            elif self.joinType == constants.TEMP_STD_POOL:
                x = Pooling.stdTemporalPool(x, self.numDiPerVideos, self.convLayers)
                # print('max pooling: ',x.size())
                x = self.bn(x)
                x = self.AdaptiveAvgPool2d(x)
                # print('AdaptiveAvgPool2d out: ', x.size())
                x = torch.flatten(x, 1)
            elif self.joinType == constants.MULT_TEMP_POOL:
                z = Pooling.stdTemporalPool(x, self.numDiPerVideos, self.convLayers) #bs,512,7,7
                y = Pooling.avgTemporalPool(x, self.numDiPerVideos, self.convLayers)
                x = Pooling.maxTemporalPool(x, self.convLayers)

                # x = self.bn(x)
                # x = self.AdaptiveAvgPool2d(x)

                # y = self.bn(y)
                # y = self.AdaptiveAvgPool2d(y)

                # z = self.bn(z)
                # z = self.AdaptiveAvgPool2d(z)
                
                # ll = [torch.flatten(x, 1), torch.flatten(y, 1), torch.flatten(z, 1)]
                # x = torch.cat(ll,dim=1)
                # print('cat poolings: ',ll.size())
                # flat = torch.flatten(x, 1) #  torch.Size([8, 25088])
                # print('flat feature map: ', flat.size())

                x = torch.add(x,z)
                x = torch.add(x, y)
                x = self.bn(x)
                x = self.AdaptiveAvgPool2d(x)
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