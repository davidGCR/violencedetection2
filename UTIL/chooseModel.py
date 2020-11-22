import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
# from MODELS.AlexNet import AlexNet
# import MODELS.ResNet as resnet
import MODELS.Vgg as vgg
from MODELS.ViolenceModels import AlexNet, AlexNetV2, ResNet, Densenet, AlexNetConv, ResNetConv, ResnetXt, ResNet_ROI_Pool
from MODELS.c3d import C3D, C3D_bn, C3D_roi_pool
from MODELS.c3d_v2 import  C3D_v2

def initializeTransferModel(model_name, num_classes, feature_extract, numDiPerVideos, joinType, classifier_file):
    # if model_name == "resnet18" or model_name == "resnet34":
    model = torch.load(classifier_file)
    model = model.cuda()
    model.inferenceMode(numDiPerVideos)
    model.enableTransferLearning(feature_extract)
    return model

def initialize_model(model_name, num_classes, freezeConvLayers, numDiPerVideos, joinType, pretrained):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    # print('feature_extract: ', feature_extract)
    model_ft = None
    input_size = 0
    if model_name == "alexnet" or model_name == "alexnetv2":
        if model_name == "alexnet":
            model_ft = AlexNet(num_classes, numDiPerVideos, joinType, freezeConvLayers)
        elif  model_name== "alexnetv2":
            model_ft = AlexNetV2(num_classes, numDiPerVideos, joinType, freezeConvLayers)
        # set_parameter_requires_grad(model_ft, feature_extract)
        input_size = 224
    elif model_name == "resnet18" or model_name == "resnet34" or model_name == "resnet50":
        # model_ft = resnet.ViolenceModelResNet(num_classes, numDiPerVideos, model_name, joinType, feature_extract)
        model_ft = ResNet(num_classes, numDiPerVideos, model_name, joinType, freezeConvLayers)
        input_size = 224
    elif model_name == "resnet-roi-pool":
        # model_ft = resnet.ViolenceModelResNet(num_classes, numDiPerVideos, model_name, joinType, feature_extract)
        model_ft = ResNet_ROI_Pool(num_classes, numDiPerVideos, model_name, joinType, freezeConvLayers)
        input_size = 224
    elif model_name == "resnetxt":
        # model_ft = resnet.ViolenceModelResNet(num_classes, numDiPerVideos, model_name, joinType, feature_extract)
        model_ft = ResnetXt(num_classes, numDiPerVideos, joinType, freezeConvLayers)
        input_size = 224
    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = Densenet(num_classes, numDiPerVideos, joinType, freezeConvLayers)
        input_size = 224
    elif model_name == "c3d":
        # model_ft = Densenet(num_classes, numDiPerVideos, joinType, freezeConvLayers)
        model_ft = C3D(pretrained=pretrained)
        model_ft.init_weights()
        input_size = 224
    elif model_name == "c3d_bn":
        # model_ft = Densenet(num_classes, numDiPerVideos, joinType, freezeConvLayers)
        model_ft = C3D_bn(pretrained=pretrained)
        model_ft.init_weights()
        input_size = 224
    elif model_name == "C3D_roi_pool":
        # model_ft = Densenet(num_classes, numDiPerVideos, joinType, freezeConvLayers)
        model_ft = C3D_roi_pool(pretrained=pretrained)
        model_ft.init_weights()
        input_size = 224
    elif model_name == "c3d_v2":
        # model_ft = Densenet(num_classes, numDiPerVideos, joinType, freezeConvLayers)
        model_ft = C3D_v2(num_classes=num_classes, pretrained=pretrained)
        # model_ft.init_weights()
        input_size = 224
    # elif model_name == "efficientnet":
    #     model_ft = MyEfficientNet(num_classes, numDiPerVideos, joinType, freezeConvLayers)
    #     input_size = 224
    # else:
    #     print("Invalid model name...")
        # exit()

    return model_ft, input_size

def initialize_FCNN(model_name, original_model):
    model = None
    if model_name == "alexnetv2":
        model = AlexNetConv(original_model=original_model)
    elif model_name == "resnet18" or model_name == "resnet34" or model_name == "resnet50":
        model = ResNetConv(original_model=original_model)
    return model
