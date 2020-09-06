import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from include import root, enviroment
import constants
import torch

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
from torch.autograd import Variable
from torch.optim import lr_scheduler
import torchvision
import time
import argparse
import copy
from tqdm import tqdm
import scipy.io as sio

import random
from transforms import hockeyTransforms, vifTransforms, ucf2CrimeTransforms, rwf_2000_Transforms
from UTIL.chooseModel import initialize_model, initialize_FCNN
from UTIL.parameters import verifiParametersToTrain
import UTIL.trainer as trainer
import UTIL.tester as tester
# from UTIL.kfolds import k_folds
from UTIL.util import  read_file, expConfig
from constants import DEVICE
# from UTIL.resultsPolicy import ResultPolicy
from UTIL.earlyStop import EarlyStopping
import pandas as pd
# from FPS import FPSMeter
from torch.utils.tensorboard import SummaryWriter
from datasetsMemoryLoader import hockeyLoadData, hockeyTrainTestSplit, vifLoadData, crime2localLoadData, customize_kfold, rwf_load_data
# from dataloader import MyDataloader
from VIOLENCE_DETECTION.violenceDataset import ViolenceDataset
import csv
from operator import itemgetter

def train_model(model, dataloaders, criterion, optimizer, num_epochs, patience, fold, path, model_config):
    since = time.time()
    val_acc_history = []
    # best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=path, model_config=model_config)

    for epoch in range(1,num_epochs+1,1):
        print('Fold ({})--Epoch {}/{}'.format(fold, epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        last_train_loss = np.inf
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for data in tqdm(dataloaders[phase]):
                inputs, labels, v_names, _, _ = data
                # print('inputs=', inputs.size(), type(inputs))
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                # zero the parameter gradients
                optimizer.zero_grad()
                batch_size = inputs.size()[0]
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)


            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            if phase == 'val':
                early_stopping(epoch_loss, epoch_acc, last_train_loss, epoch, fold, model)
            else:
                last_train_loss = epoch_loss
        if early_stopping.early_stop:
            print("Early stopping")
            break
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    return model, early_stopping.best_acc, early_stopping.val_loss_min, early_stopping.best_epoch

def base_dataset(dataset, mean=None, std=None):
    if dataset == 'ucfcrime2local':
        datasetAll, labelsAll, numFramesAll = crime2localLoadData(min_frames=40)
        transforms = ucf2CrimeTransforms(224, mean=mean, std=std)
    elif dataset == 'vif':
        datasetAll, labelsAll, numFramesAll, splitsLen = vifLoadData(constants.PATH_VIF_FRAMES)
        transforms = vifTransforms(input_size=224, mean=mean, std=std)
    elif dataset == 'hockey':
        datasetAll, labelsAll, numFramesAll = hockeyLoadData(shuffle=True)
        transforms = hockeyTransforms(input_size=224, mean=mean, std=std)
    elif dataset == 'rwf-2000':
        train_names, train_labels, train_num_frames, test_names, test_labels, test_num_frames = rwf_load_data()
        transforms = rwf_2000_Transforms(input_size=224, mean=mean, std=std)
        return train_names, train_labels, train_num_frames, test_names, test_labels, test_num_frames, transforms
    
    return datasetAll, labelsAll, numFramesAll, transforms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelType", type=str, default="alexnet", help="model")
    parser.add_argument("--dataset",type=str)
    parser.add_argument("--numEpochs",type=int,default=30)
    parser.add_argument("--batchSize",type=int,default=64)
    parser.add_argument("--freezeConvLayers",type=lambda x: (str(x).lower() == 'true'), default=False, help="to fine tunning")
    parser.add_argument("--numDynamicImagesPerVideo", type=int)
    parser.add_argument("--joinType", type=str)
    parser.add_argument("--foldsNumber", type=int, default=5)
    parser.add_argument("--numWorkers", type=int, default=4)
    parser.add_argument("--videoSegmentLength", type=int)
    parser.add_argument("--positionSegment", type=str)
    parser.add_argument("--split_type", type=str)
    parser.add_argument("--overlapping", type=float)
    parser.add_argument("--frameSkip", type=int, default=0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--skipInitialFrames", type=int, default=0)
    parser.add_argument("--transferModel", type=str, default=None)
    parser.add_argument("--saveCheckpoint", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--useKeyframes", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--windowLen", type=int, default=0)
    # parser.add_argument("--segmentPreprocessing", type=lambda x: (str(x).lower() == 'true'), default=False)

    args = parser.parse_args()
    input_size = 224
    shuffle = True
    if args.dataset == 'rwf-2000':
        datasetAll = []
    else:
        datasetAll, labelsAll, numFramesAll, transforms = base_dataset(args.dataset)
    cv_test_accs = []
    cv_test_losses = []
    cv_final_epochs = []
    patience = 5
    folds_number = 5
    fold = 0
    checkpoint_path = None
    config = None
    print(args.dataset)
       
    for train_idx, test_idx in customize_kfold(n_splits=folds_number, dataset=args.dataset, X_len=len(datasetAll), shuffle=shuffle):
        fold = fold + 1
        print("**************** Fold:{}/{} ".format(fold, folds_number))
        if args.dataset == 'rwf-2000':
            train_x, train_y, train_numFrames, test_x, test_y, test_numFrames, transforms = base_dataset(args.dataset)
            print(len(train_x), len(train_y), len(train_numFrames), len(test_x), len(test_y), len(test_numFrames))
        else:
            train_x, train_y, test_x, test_y = None, None, None, None
            train_x = list(itemgetter(*train_idx)(datasetAll))
            train_y = list(itemgetter(*train_idx)(labelsAll))
            train_numFrames = list(itemgetter(*train_idx)(numFramesAll))
            test_x = list(itemgetter(*test_idx)(datasetAll))
            test_y = list(itemgetter(*test_idx)(labelsAll))
            test_numFrames = list(itemgetter(*test_idx)(numFramesAll))

        train_dataset = ViolenceDataset(dataset=train_x,
                                        labels=train_y,
                                        numFrames=train_numFrames,
                                        spatial_transform=transforms['train'],
                                        numDynamicImagesPerVideo=args.numDynamicImagesPerVideo,
                                        videoSegmentLength=args.videoSegmentLength,
                                        positionSegment=args.positionSegment,
                                        overlaping=args.overlapping,
                                        frame_skip=args.frameSkip,
                                        skipInitialFrames=args.skipInitialFrames,
                                        ppType=None,
                                        useKeyframes=args.useKeyframes,
                                        windowLen=args.windowLen)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=True, num_workers=args.numWorkers)

        test_dataset = ViolenceDataset(dataset=test_x,
                                        labels=test_y,
                                        numFrames=test_numFrames,
                                        spatial_transform=transforms['val'],
                                        numDynamicImagesPerVideo=args.numDynamicImagesPerVideo,
                                        videoSegmentLength=args.videoSegmentLength,
                                        positionSegment=args.positionSegment,
                                        overlaping=args.overlapping,
                                        frame_skip=args.frameSkip,
                                        skipInitialFrames=args.skipInitialFrames,
                                        ppType=None,
                                        useKeyframes=args.useKeyframes,
                                        windowLen=args.windowLen)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batchSize, shuffle=True, num_workers=args.numWorkers)
        
        dataloaders = {'train': train_dataloader, 'val': test_dataloader}

        model, _ = initialize_model(model_name=args.modelType,
                                    num_classes=2,
                                    freezeConvLayers=args.freezeConvLayers,
                                    numDiPerVideos=args.numDynamicImagesPerVideo,
                                    joinType=args.joinType,
                                    use_pretrained=True)
        model.to(DEVICE)
        params_to_update = verifiParametersToTrain(model, args.freezeConvLayers, printLayers=True)
        # print(params_to_update)
        optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        criterion = nn.CrossEntropyLoss()
        
        # for (key, val) in config.items():
        #     if key != '\'log_dir\'':
        #         ss.join("{!s}={!r}".format(key, val))
        # log_dir = os.path.join(constants.PATH_RESULTS, args.dataset.upper(), 'tensorboard-runs', experimentConfig_str)
        # writer = SummaryWriter(log_dir)
        if args.saveCheckpoint:
            config = {
                'dataset':args.dataset,
                'model': args.modelType,
                'numEpochs': args.numEpochs,
                'featureExtract': args.featureExtract,
                'numDynamicImages':args.numDynamicImagesPerVideo,
                'segmentLength':args.videoSegmentLength,
                'frameSkip':args.frameSkip,
                'skipInitialFrames':args.skipInitialFrames,
                'overlap':args.overlapping,
                'joinType': args.joinType,
                'log_dir': None,
                'useKeyframes': args.useKeyframes,
                'windowLen': args.windowLen
            }    
            # ss = '_'.join("{!s}={!r}".format(key, val) for (key, val) in config.items())
            # ss = ss+'_fold='+str(fold)
            ss = ""
            for (key, val) in config.items():
                if key != 'log_dir':
                    ss = ss + "_{!s}={!r}".format(key, val)
            ss = ss.replace("\'", "")
            # print(ss)
            checkpoint_path = os.path.join(constants.PATH_RESULTS, args.dataset.upper(), 'checkpoints', 'DYN_Stream-{}-fold={}'.format(ss,fold))
            
        model, best_acc, val_loss_min, best_epoch = train_model(model,
                                                            dataloaders,
                                                            criterion,
                                                            optimizer,
                                                            num_epochs=args.numEpochs,
                                                            patience=patience,
                                                            fold=fold,
                                                            path=checkpoint_path,
                                                            model_config=config)
        cv_test_accs.append(best_acc.item())
        cv_test_losses.append(val_loss_min)
        cv_final_epochs.append(best_epoch)


    print('CV Accuracies=', cv_test_accs)
    print('CV Losses=', cv_test_losses)
    print('CV Epochs=', cv_final_epochs)
    print('Test AVG Accuracy={}, Test AVG Loss={}'.format(np.average(cv_test_accs), np.average(cv_test_losses)))
    print("Accuracy: %0.3f (+/- %0.3f), Losses: %0.3f" % (np.array(cv_test_accs).mean(), np.array(cv_test_accs).std() * 2, np.array(cv_test_losses).mean()))


if __name__ == "__main__":
    main()
    # key_frame_selection()