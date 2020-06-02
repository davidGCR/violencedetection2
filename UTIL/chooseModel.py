import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
# from MODELS.AlexNet import AlexNet
# import MODELS.ResNet as resnet
import MODELS.Vgg as vgg
from MODELS.ViolenceModels import AlexNet, AlexNetV2, ResNet, Densenet

def initializeTransferModel(model_name, num_classes, feature_extract, numDiPerVideos, joinType, classifier_file):
    # if model_name == "resnet18" or model_name == "resnet34":
    model = torch.load(classifier_file)
    model = model.cuda()
    model.inferenceMode(numDiPerVideos)
    model.enableTransferLearning(feature_extract)
    return model

def initialize_model(model_name, num_classes, feature_extract, numDiPerVideos, joinType, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0
    if model_name == "alexnet" or model_name == "alexnetv2":
        if model_name == "alexnet":
            model_ft = AlexNet(num_classes, numDiPerVideos, joinType, feature_extract)
        elif  model_name== "alexnetv2":
            model_ft = AlexNetV2(num_classes, numDiPerVideos, joinType, feature_extract)
        # set_parameter_requires_grad(model_ft, feature_extract)
        input_size = 224
    elif model_name == "resnet18" or model_name == "resnet34" or model_name == "resnet50":
        # model_ft = resnet.ViolenceModelResNet(num_classes, numDiPerVideos, model_name, joinType, feature_extract)
        model_ft = ResNet(num_classes, numDiPerVideos, model_name, joinType, feature_extract)
        input_size = 224
    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = Densenet(num_classes, numDiPerVideos, joinType, feature_extract)
        
        input_size = 224
    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = vgg.ViolenceModelVGG(numDiPerVideos, model_name, joinType, feature_extract)
        input_size = 224
    else:
        print("Invalid model name...")
        # exit()

    return model_ft, input_size