import torch.nn as nn
from torchvision import models
# from util import *
from UTIL.parameters import set_parameter_requires_grad
# from tempPooling import *
import MODELS.Pooling as Pooling
from MODELS.Identity import Identity
import torch
import constants


class AlexNet(nn.Module):  # ViolenceModel2
    def __init__(self, num_classes, numDiPerVideos, joinType, feature_extract):
        super(AlexNet, self).__init__()
        self.numDiPerVideos = numDiPerVideos
        self.joinType = joinType
        self.num_classes = num_classes
        self.model = models.alexnet(pretrained=True)
        set_parameter_requires_grad(self.model, feature_extract)
        self.linear = None
        self.model = nn.Sequential(*list(self.model.children())[:-2])  # to tempooling
        self.tmpPooling = nn.MaxPool2d((numDiPerVideos, 1))
        self.linear = nn.Linear(256 * 6 * 6, self.num_classes)
        # num_ftrs = model_ft.classifier[6].in_features
        # self.model.classifier[6] = nn.Linear(num_ftrs,num_classes)

    def forward(self, x):
        # numDynamicImages = x.size(0)
        # lista = []
        # x = x.permute(1, 0, 2, 3, 4)
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

class AlexNetV2(nn.Module):  # ViolenceModel2
    def __init__(self, num_classes, numDiPerVideos, joinType, feature_extract):
        super(AlexNetV2, self).__init__()
        self.numDiPerVideos = numDiPerVideos
        self.joinType = joinType
        self.num_classes = num_classes
        self.model = models.alexnet(pretrained=True)
        set_parameter_requires_grad(self.model, feature_extract)
        self.linear = None
        self.model = nn.Sequential(*list(self.model.children())[:-2])  # to tempooling
        
        
        self.tmpPooling = nn.MaxPool2d((numDiPerVideos, 1))

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        # print(batch_size, timesteps, C, H, W)
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
        x = self.classifier(x)
        return x

class AlexNetConv(nn.Module):
    def __init__(self, original_model):
        super(AlexNetConv, self).__init__()
        self.features = nn.Sequential(
            # stop at conv4
            *list(original_model.model.children())
        )
    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        x = x.view(batch_size * timesteps, C, H, W)
        x = self.features(x)
        return x

class ResNetConv(nn.Module):
    def __init__(self, original_model):
        super(ResNetConv, self).__init__()
        self.convLayers = nn.Sequential(
            *list(original_model.convLayers.children())
        )

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        x = x.view(batch_size * timesteps, C, H, W)
        x = self.convLayers(x)
        return x

class ResNet(nn.Module):
    def __init__(self, num_classes, numDiPerVideos, model_name, joinType ,feature_extract):
        super(ResNet, self).__init__()
        self.numDiPerVideos = numDiPerVideos
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
        if self.joinType == constants.TEMP_MAX_POOL or self.joinType == constants.MULT_TEMP_POOL or self.joinType == constants.TEMP_AVG_POOL or self.joinType == constants.TEMP_STD_POOL:
            if model_name == 'resnet18' or model_name == 'resnet34':
                self.linear = nn.Linear(512, self.num_classes)
            elif model_name == 'resnet50':
                self.linear = nn.Linear(2048, self.num_classes)
            # if model_name == 'resnet18' else nn.Linear(512*7*7,self.num_classes)

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        x = self.convLayers(c_in)
        x = self.AdaptiveAvgPool2d(x)
        # print('conv: ', x.size())
        x = torch.flatten(x, 1)
        num_fc_input_features = self.linear.in_features
        x = x.view(batch_size, timesteps, num_fc_input_features)
        x = x.max(dim=1).values
        # x = self.bn(x)
        
        # x = torch.flatten(x, 1)
        x = self.linear(x)
        return x


class Densenet(nn.Module):  # ViolenceModel2
    def __init__(self, num_classes, numDiPerVideos, joinType, feature_extract):
        super(Densenet, self).__init__()
        self.numDiPerVideos = numDiPerVideos
        self.joinType = joinType
        self.num_classes = num_classes
        self.model = models.densenet121(pretrained=True)
        set_parameter_requires_grad(self.model, feature_extract)
        self.num_ftrs = self.model.classifier.in_features
        self.model.classifier = Identity()
        self.linear= nn.Linear(self.num_ftrs, num_classes)
        
        # self.tmpPooling = nn.MaxPool2d((numDiPerVideos, 1))

    def forward(self, x):
        # print(x.size())
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        # print('cin: ', c_in.size())
        c_out = self.model(c_in)
        # print('cout: ', c_out.size())
        x = torch.flatten(c_out, 1)
        # print('flatten: ', x.size())
        # Re-structure the data and then temporal max-pool.
        x = x.view(batch_size, timesteps, self.num_ftrs)
        # print('Re-structure: ', x.size())
        x = x.max(dim=1).values
        # print('maxpooling: ', x.size())
        x = self.linear(x)
        return x

class ResnetXt(nn.Module):  # ViolenceModel2
    def __init__(self, num_classes, numDiPerVideos, joinType, feature_extract):
        super(ResnetXt, self).__init__()
        # Define the ResNet.
        resnet = models.resnext50_32x4d(pretrained=True)
        resnet.fc = nn.Sequential()

        # Define the classifier.
        self.features = resnet
        self.fc = nn.Sequential(nn.Linear(2048, 512),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=0.5),
                                nn.Linear(512, num_classes))

    def forward(self, x):
        batch_size, segment_size, c, h, w = x.shape
        num_fc_input_features = self.fc[0].in_features

        # Time distribute the inputs.
        x = x.view(batch_size * segment_size, c, h, w)
        x = self.features(x)

        # Re-structure the data and then temporal max-pool.
        x = x.view(batch_size, segment_size, num_fc_input_features)
        x = x.max(dim=1).values

        # FC.
        x = self.fc(x)
        return x

class Inception(nn.Module):  # ViolenceModel2
    def __init__(self, num_classes, numDiPerVideos, joinType, feature_extract):
        super(Inception, self).__init__()
        self.numDiPerVideos = numDiPerVideos
        self.joinType = joinType
        self.num_classes = num_classes
        self.model = models.inception_v3(pretrained=True)
        set_parameter_requires_grad(self.model, feature_extract)
         # Handle the auxilary net
        num_ftrs = self.model.AuxLogits.fc.in_features
        self.model.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        
        # self.num_ftrs = self.model.classifier.in_features
        # self.model.classifier = Identity()
        # self.linear= nn.Linear(self.num_ftrs, num_classes)
        

    def forward(self, x):
        # print(x.size())
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        # print('cin: ', c_in.size())
        c_out = self.model(c_in)
        # print('cout: ', c_out.size())
        x = torch.flatten(c_out, 1)
        # print('flatten: ', x.size())
        # Re-structure the data and then temporal max-pool.
        x = x.view(batch_size, timesteps, self.num_ftrs)
        # print('Re-structure: ', x.size())
        x = x.max(dim=1).values
        # print('maxpooling: ', x.size())
        x = self.linear(x)
        return x

from efficientnet_pytorch import EfficientNet

class MyEfficientNet(nn.Module):  # ViolenceModel2
    def __init__(self, num_classes, numDiPerVideos, joinType, feature_extract):
        super(MyEfficientNet, self).__init__()
        self.numDiPerVideos = numDiPerVideos
        self.joinType = joinType
        self.num_classes = num_classes
        self.model = EfficientNet.from_name('efficientnet-b1')
        set_parameter_requires_grad(self.model, feature_extract)
        self.num_ftrs = self.model._fc.in_features

        self.model._fc=Identity()
        # self.convLayers = nn.Sequential(*list(self.model.children())[:-5])
        # self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        # self._dropout = nn.Dropout(p=0.2, inplace=False)
        self.linear = nn.Linear(self.num_ftrs, 2)  
        

    def forward(self, x):
        # print(x.size())
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        # print('cin: ', c_in.size())

        x = self.model(c_in)

        x = torch.flatten(x, 1)
        x = x.view(batch_size, timesteps, self.num_ftrs)
        x = x.max(dim=1).values

        # x = self._avg_pooling(x)
        # x = x.view(bs, -1)
        # x = self._dropout(x)
        x = self.linear(x)

        return x