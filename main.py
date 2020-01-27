import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import glob
import cv2
from PIL import Image
from torch.utils.data import Dataset
from torchvision import models
from torchvision import transforms
from PIL import Image
from torch.autograd import Variable
# from tensorboardcolab import TensorBoardColab
import time
from torch.optim import lr_scheduler
import argparse
import sys

# sys.path.insert(1, "/media/david/datos/PAPERS-SOURCE_CODE/MyCode")
sys.path.insert(1,'/Users/davidchoqueluqueroman/Desktop/PAPERS-CODIGOS/violencedetection2')
from AlexNet import *
from ViolenceDatasetV2 import *
from trainer import *
from kfolds import *
from operator import itemgetter
import random
from initializeModel import *
from util import *
from parameters import *
from transforms import *
from MaskDataset import MaskDataset
from saliency_model import *
import initializeDataset
import constants



def init(dataset, vif_path, hockey_path_violence, hockey_path_noviolence, path_learning_curves, path_checkpoints, modelType, ndis, num_workers, data_transforms, dataset_source, interval_duration, avgmaxDuration,
    batch_size, num_epochs, feature_extract, joinType, scheduler_type, device, criterion, folds_number, debugg_mode = False, salModelFile=''):
    
    for numDiPerVideos in ndis: #for experiments
        train_lost = []
        train_acc = []
        test_lost = []
        test_acc = []
        if dataset == 'hockey' or  dataset == 'masked':
            datasetAll, labelsAll, numFramesAll = initializeDataset.createDataset(hockey_path_violence, hockey_path_noviolence) #shuffle
            # combined = list(zip(datasetAll, labelsAll, numFramesAll))
            # random.shuffle(combined)
            # datasetAll[:], labelsAll[:], numFramesAll[:] = zip(*combined) 
            # train_idx, test_idx = None, None
        # elif dataset == 'violentflows':

        print("CONFIGURATION: ", "modelType:", modelType, ", numDiPerVideos:", numDiPerVideos,
                ", dataset_source:", dataset_source, ", batch_size:", batch_size, ", num_epochs:",
                num_epochs, ", feature_extract:", feature_extract, ", joinType:", joinType, ", scheduler_type: ", scheduler_type, )

        
        fold = 0
        for train_idx, test_idx in k_folds(n_splits=folds_number, subjects=len(datasetAll)):
        # for dataset_train, dataset_train_labels,dataset_test,dataset_test_labels   in k_folds_from_folders(vif_path, 5):
            fold = fold + 1
            print("**************** Fold:{}/{} ".format(fold, folds_number))
            train_x, train_y, test_x, test_y = None, None, None, None
            if dataset == 'hockey' or dataset == 'masked':
                print('fold: ',len(train_idx),len(test_idx))
                train_x = list(itemgetter(*train_idx)(datasetAll))
                train_y = list(itemgetter(*train_idx)(labelsAll))
                test_x = list(itemgetter(*test_idx)(datasetAll))
                test_y = list(itemgetter(*test_idx)(labelsAll))
                initialize_dataset.print_balance(train_y, test_y)
                

            dataloaders_dict = initialize_dataset.getDataLoaders(dataset, train_x, train_y, test_x, test_y, data_transforms, numDiPerVideos, dataset_source,
                                                avgmaxDuration, interval_duration, batch_size, num_workers, debugg_mode, salModelFile)
            
            model = None
            model, input_size = initialize_model( model_name=modelType, num_classes=2, feature_extract=feature_extract, numDiPerVideos=numDiPerVideos, joinType=joinType, use_pretrained=True)
            model.to(device)
            #only print parameters to train
            params_to_update = verifiParametersToTrain(model, feature_extract)
            # Observe that all parameters are being optimized
            optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
            # Decay LR by a factor of 0.1 every 7 epochs
            if scheduler_type == "StepLR":
                exp_lr_scheduler = lr_scheduler.StepLR( optimizer, step_size=7, gamma=0.1 )
            elif scheduler_type == "OnPlateau":
                exp_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau( optimizer, patience=5, verbose=True )
            ### trainer
            MODEL_NAME = get_model_name(modelType, scheduler_type, numDiPerVideos, dataset_source, feature_extract, joinType)
            if folds_number == 1:
                MODEL_NAME = MODEL_NAME+constants.LABEL_PRODUCTION_MODEL
            print('model_name: ', MODEL_NAME)

            if folds_number == 1:
                trainer = Trainer(model, dataloaders_dict, criterion, optimizer, exp_lr_scheduler, device, num_epochs, os.path.join(path_checkpoints, MODEL_NAME),numDiPerVideos)
            else:
                trainer = Trainer( model, dataloaders_dict, criterion, optimizer, exp_lr_scheduler, device, num_epochs, None, numDiPerVideos)

            for epoch in range(1, num_epochs + 1):
                print("----- Epoch {}/{}".format(epoch, num_epochs))
                # Train and evaluate
                epoch_loss_train, epoch_acc_train = trainer.train_epoch(epoch)
                epoch_loss_test, epoch_acc_test = trainer.test_epoch(epoch)
                exp_lr_scheduler.step(epoch_loss_test)
                train_lost.append(epoch_loss_train)
                train_acc.append(epoch_acc_train)
                test_lost.append(epoch_loss_test)
                test_acc.append(epoch_acc_test)
            #   tb.save_value("trainLoss", "train_loss", foldidx*num_epochs + epoch, epoch_loss_train)
            #   tb.save_value("trainAcc", "train_acc", foldidx*num_epochs + epoch, epoch_acc_train)
            #   tb.save_value("testLoss", "test_loss", foldidx*num_epochs + epoch, epoch_loss_test)
            #   tb.save_value("testAcc", "test_acc", foldidx*num_epochs + epoch, epoch_acc_test)

            #   tb.flush_line('train_loss')
            #   tb.flush_line('train_acc')
            #   tb.flush_line('test_loss')
            #   tb.flush_line('test_acc')

            #     filepath = path_models+str(modelType)+'('+str(numDiPerVideos)+'di)-fold-'+str(foldidx)+'.pt'
            #     torch.save({
            #         'kfold': foldidx,
            #         'model_state_dict': model.state_dict(),
            #         'optimizer_state_dict': optimizer.state_dict()
            #         }, filepath)
        print("saving loss and acc history...")
        saveLearningCurve(os.path.join(path_learning_curves,MODEL_NAME+"-train_lost.txt"), train_lost)
        saveLearningCurve(os.path.join(path_learning_curves,MODEL_NAME+"-train_acc.txt"), train_acc)
        saveLearningCurve(os.path.join(path_learning_curves,MODEL_NAME+"-test_lost.txt"), test_lost)
        saveLearningCurve(os.path.join(path_learning_curves,MODEL_NAME+"-test_acc.txt"),test_acc)
        # saveList(path_learning_curves, modelType, scheduler_type, "train_lost", numDiPerVideos, dataset_source, feature_extract, joinType, train_lost,)
        # saveList(path_learning_curves, modelType, scheduler_type,"train_acc",numDiPerVideos, dataset_source, feature_extract, joinType, train_acc, )
        # saveList(path_learning_curves, modelType, scheduler_type, "test_lost", numDiPerVideos, dataset_source, feature_extract, joinType, test_lost, )
        # saveList(path_learning_curves, modelType, scheduler_type, "test_acc", numDiPerVideos, dataset_source, feature_extract, joinType, test_acc, )

def __main__():

    # python3 main.py --dataset hockey --numEpochs 12 --ndis 1 --foldsNumber 1 --featureExtract true --checkpointPath BlackBoxModels
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",type=str)
    parser.add_argument("--vifPath",type=str,default=constants.PATH_VIOLENTFLOWS_FRAMES,help="Directory for Violent Flows dataset")
    parser.add_argument("--pathViolence",type=str,default=constants.PATH_HOCKEY_FRAMES_VIOLENCE,help="Directory containing violence videos")
    parser.add_argument("--pathNonviolence",type=str,default=constants.PATH_HOCKEY_FRAMES_NON_VIOLENCE,help="Directory containing non violence videos")
    parser.add_argument("--pathLearningCurves", type=str, default=constants.PATH_LEARNING_CURVES_DI, help="Directory containing results")
    parser.add_argument("--checkpointPath", type=str, default=constants.PATH_CHECKPOINTS_DI)
    parser.add_argument("--modelType",type=str,default="alexnet",help="model")
    parser.add_argument("--numEpochs",type=int,default=30)
    parser.add_argument("--batchSize",type=int,default=64)
    parser.add_argument("--featureExtract",type=lambda x: (str(x).lower() == 'true'), default=False, help="to fine tunning")
    parser.add_argument("--schedulerType",type=str,default="OnPlateau",help="learning rate scheduler")
    parser.add_argument("--debuggMode", type=bool, default=False, help="show prints")
    parser.add_argument("--ndis", nargs='+', type=int, help="num dyn imgs")
    parser.add_argument("--joinType", type=str, default="tempMaxPool", help="show prints")
    parser.add_argument("--foldsNumber", type=int, default=5)
    parser.add_argument("--salModelFile", type=str, default='')
    parser.add_argument("--numWorkers", type=int, default=4)

    args = parser.parse_args()
    dataset = args.dataset
    vif_path = args.vifPath
    path_learning_curves = args.pathLearningCurves
    path_violence = args.pathViolence
    path_noviolence = args.pathNonviolence
    modelType = args.modelType
    batch_size = args.batchSize
    num_epochs = args.numEpochs
    feature_extract = args.featureExtract
    joinType = args.joinType
    scheduler_type = args.schedulerType
    debugg_mode = args.debuggMode
    ndis = args.ndis
    path_checkpoints = args.checkpointPath
    folds_number = args.foldsNumber
    dataset_source = "frames"
    debugg_mode = False
    avgmaxDuration = 1.66
    interval_duration = 0.3
    num_workers = args.numWorkers
    input_size = 224
    salModelFile = args.salModelFile

    transforms = createTransforms(input_size)
    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()
    # path_models = '/media/david/datos/Violence DATA/violentflows/Models/'
    # path_results = '/media/david/datos/Violence DATA/violentflows/Results/'+dataset_source
    # gpath = '/media/david/datos/Violence DATA/violentflows/movies Frames'
    init(dataset, vif_path, path_violence, path_noviolence, path_learning_curves, path_checkpoints, modelType, ndis, num_workers, transforms,
            dataset_source, interval_duration, avgmaxDuration, batch_size, num_epochs, feature_extract, joinType, scheduler_type,
            device, criterion, folds_number, debugg_mode, salModelFile)

__main__()

