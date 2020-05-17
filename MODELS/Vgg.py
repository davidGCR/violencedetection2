import torch.nn as nn
from torchvision import models
# from util import * 
from MODELS.Identity import Identity
import MODELS.Pooling as Pooling
import torch

class ViolenceModelVGG(nn.Module):
    def __init__(self, seqLen, model_name, joinType ,feature_extract):
        super(ViolenceModelVGG, self).__init__()
        self.seqLen = seqLen
        self.joinType = joinType
        
        self.model_ft = models.vgg11_bn(pretrained=True)
        self.num_ftrs = self.model_ft.classifier[6].in_features
        # self.model_ft.fc = Identity()
        # self.convLayers = nn.Sequential(*list(self.model_ft.children())[:-2]) # to tempooling

        set_parameter_requires_grad(self.model_ft, feature_extract)
        if self.joinType == 'cat':
            self.model_ft.classifier[3] = nn.Linear(4096,2048)
            self.model_ft.classifier = self.model_ft.classifier[:-1]
            # self.linear = nn.Linear(self.num_ftrs*self.seqLen,2)
            self.linear = nn.Linear(2048*self.seqLen,2)
        elif self.joinType == 'tempMaxPool':
            self.model_ft = nn.Sequential(*list(self.model_ft.children())[:-2]) #remove fc layers
            self.linear = nn.Linear(512*7*7,2)
    
    def forward(self, x):
        # print('forward input size:',x.size())
        if self.joinType == 'cat':
            x = self.getFeatureVectorCat(x)
            # print('cat input size:',x.size())
        elif self.joinType == 'tempMaxPool':
            x = self.getFeatureVectorTempPool(x)
            # print('tempPooling input size:',x.size())
        # print('linear input size:',x.size())
        x = self.linear(x)
        return x
        
    def getFeatureVectorTempPool(self, x):
        lista = []
        for dimage in range(0, self.seqLen):
            feature = self.model_ft(x[dimage])
            lista.append(feature)

        minibatch = torch.stack(lista, 0)
        minibatch = minibatch.permute(1, 0, 2, 3, 4)
        num_dynamic_images = self.seqLen
        tmppool = nn.MaxPool2d((num_dynamic_images, 1))
        lista_minibatch = []
        for idx in range(minibatch.size()[0]):
            out = tempMaxPooling(minibatch[idx], tmppool)
            lista_minibatch.append(out)

        feature = torch.stack(lista_minibatch, 0)
        feature = torch.flatten(feature, 1)
        return feature
    
    def getFeatureVectorCat(self, x):
        lista = []
        for dimage in range(0, self.seqLen):
            feature = self.model_ft(x[dimage])
            lista.append(feature)
        x = torch.cat(lista, dim=1)
        return x