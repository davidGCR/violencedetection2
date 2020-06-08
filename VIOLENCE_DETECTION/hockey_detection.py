

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

import random
from transforms import hockeyTransforms
from UTIL.chooseModel import initialize_model
from UTIL.parameters import verifiParametersToTrain
import UTIL.trainer as trainer
import UTIL.tester as tester
from UTIL.kfolds import k_folds
from UTIL.util import  read_file
from constants import DEVICE
from UTIL.resultsPolicy import ResultPolicy
import pandas as pd
# from FPS import FPSMeter
from torch.utils.tensorboard import SummaryWriter
from datasetsPreprocessing import hockeyLoadData, hockeyTrainTestSplit
from dataloader import Dataloader
import csv
from operator import itemgetter

def __main__():

    # python3 main.py --dataset hockey --numEpochs 12 --ndis 1 --foldsNumber 1 --featureExtract true --checkpointPath BlackBoxModels
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelType",type=str,default="alexnet",help="model")
    parser.add_argument("--numEpochs",type=int,default=30)
    parser.add_argument("--batchSize",type=int,default=64)
    parser.add_argument("--featureExtract",type=lambda x: (str(x).lower() == 'true'), default=False, help="to fine tunning")
    parser.add_argument("--numDynamicImagesPerVideo", type=int)
    parser.add_argument("--joinType", type=str)
    parser.add_argument("--foldsNumber", type=int, default=5)
    parser.add_argument("--numWorkers", type=int, default=4)
    parser.add_argument("--videoSegmentLength", type=int)
    parser.add_argument("--positionSegment", type=str)
    parser.add_argument("--split_type", type=str)
    parser.add_argument("--overlapping", type=float)
    parser.add_argument("--frameSkip", type=int, default=0)
    parser.add_argument("--transferModel", type=str, default=None)
    parser.add_argument("--saveCheckpoint", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--segmentPreprocessing", type=lambda x: (str(x).lower() == 'true'), default=False)

    args = parser.parse_args()
    input_size = 224
    shuffle = True
    datasetAll, labelsAll, numFramesAll = hockeyLoadData()
    transforms = hockeyTransforms(input_size)
    cv_test_accs = []
    cv_test_losses = []
    cv_final_epochs = []

    if args.split_type[:-2] == 'train-test':
        train_x, train_y, train_numFrames, test_x, test_y, test_numFrames = hockeyTrainTestSplit(args.split_type, datasetAll, labelsAll, numFramesAll)
        train_dt_loader = Dataloader(X=train_x,
                                    y=train_y,
                                    numFrames=train_numFrames,
                                    transform=transforms['train'],
                                    NDI=args.numDynamicImagesPerVideo,
                                    videoSegmentLength=args.videoSegmentLength,
                                    positionSegment=args.positionSegment,
                                    overlapping=args.overlapping,
                                    frameSkip=args.frameSkip,
                                    skipInitialFrames=0,
                                    segmentPreprocessing=args.segmentPreprocessing,
                                    batchSize=args.batchSize,
                                    shuffle=True,
                                    numWorkers=args.numWorkers)
        
        test_dt_loader = Dataloader(X=test_x,
                                    y=test_y,
                                    numFrames=test_numFrames,
                                    transform=transforms['val'],
                                    NDI=args.numDynamicImagesPerVideo,
                                    videoSegmentLength=args.videoSegmentLength,
                                    positionSegment=args.positionSegment,
                                    overlapping=args.overlapping,
                                    frameSkip=args.frameSkip,
                                    skipInitialFrames=0,
                                    segmentPreprocessing=args.segmentPreprocessing,
                                    batchSize=args.batchSize,
                                    shuffle=True,
                                    numWorkers=args.numWorkers)
            
        model, _ = initialize_model(model_name=args.modelType,
                                    num_classes=2,
                                    feature_extract=args.featureExtract,
                                    numDiPerVideos=args.numDynamicImagesPerVideo,
                                    joinType=args.joinType,
                                    use_pretrained=True)
        model.to(DEVICE)
        params_to_update = verifiParametersToTrain(model, args.featureExtract)
        optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        criterion = nn.CrossEntropyLoss()
        experimentConfig = 'HOCKEY-Model-{},segmentLen-{},numDynIms-{},frameSkip-{},segmentPreprocessing-{},epochs-{},split_type-{}'.format(args.modelType,
                                                                                                                    args.videoSegmentLength,
                                                                                                                    args.numDynamicImagesPerVideo,
                                                                                                                    args.frameSkip,
                                                                                                                    args.segmentPreprocessing,
                                                                                                                    args.numEpochs,
                                                                                                                    args.split_type)
        log_dir = os.path.join(constants.PATH_RESULTS, 'HOCKEY', 'tensorboard-runs', experimentConfig)
        writer = SummaryWriter(log_dir)
        print('Tensorboard logDir={}'.format(log_dir))
        tr = trainer.Trainer(model=model,
                            model_transfer= args.transferModel,
                            train_dataloader=train_dt_loader.getDataloader(),
                            val_dataloader=test_dt_loader.getDataloader(),
                            criterion=criterion,
                            optimizer=optimizer,
                            num_epochs=args.numEpochs,
                            checkpoint_path=os.path.join(constants.PATH_RESULTS, 'HOCKEY', 'checkpoints', experimentConfig),
                            lr_scheduler=None)
        
        final_val_loss = 1000
        final_train_loss = 1000
        final_val_acc = 0
        best_epoch = 0
        policy = ResultPolicy()

        for epoch in range(1, args.numEpochs + 1):
            print("----- Epoch {}/{}".format(epoch, args.numEpochs))
            epoch_loss_train, epoch_acc_train = tr.train_epoch(epoch)
            epoch_loss_val, epoch_acc_val = tr.val_epoch(epoch)
            exp_lr_scheduler.step()

            flac = policy.update(epoch_loss_train, epoch_acc_train, epoch_loss_val, epoch_acc_val, epoch)
            if args.saveCheckpoint:
                tr.saveCheckpoint(epoch, flac)

            writer.add_scalar('training loss', epoch_loss_train, epoch)
            writer.add_scalar('validation loss', epoch_loss_val, epoch)
            writer.add_scalar('training Acc', epoch_acc_train, epoch)
            writer.add_scalar('validation Acc', epoch_acc_val, epoch)

        cv_test_accs.append(policy.getFinalTestAcc())
        cv_test_losses.append(policy.getFinalTestLoss())
        cv_final_epochs.append(policy.getFinalEpoch())
            
    elif args.split_type == 'cross-val':
        folds_number = 5
        fold = 0
        for train_idx, test_idx in k_folds(n_splits=folds_number, subjects=len(datasetAll), splits_folder=constants.PATH_HOCKEY_README):
            fold = fold + 1
            print("**************** Fold:{}/{} ".format(fold, folds_number))
            train_x, train_y, test_x, test_y = None, None, None, None
            train_x = list(itemgetter(*train_idx)(datasetAll))
            train_y = list(itemgetter(*train_idx)(labelsAll))
            train_numFrames = list(itemgetter(*train_idx)(numFramesAll))
            test_x = list(itemgetter(*test_idx)(datasetAll))
            test_y = list(itemgetter(*test_idx)(labelsAll))
            test_numFrames = list(itemgetter(*test_idx)(numFramesAll))
            train_dt_loader = Dataloader(X=train_x,
                                        y=train_y,
                                        numFrames=train_numFrames,
                                        transform=transforms['train'],
                                        NDI=args.numDynamicImagesPerVideo,
                                        videoSegmentLength=args.videoSegmentLength,
                                        positionSegment=args.positionSegment,
                                        overlapping=args.overlapping,
                                        frameSkip=args.frameSkip,
                                        skipInitialFrames=0,
                                        segmentPreprocessing=args.segmentPreprocessing,
                                        batchSize=args.batchSize,
                                        shuffle=True,
                                        numWorkers=args.numWorkers)
        
            test_dt_loader = Dataloader(X=test_x,
                                        y=test_y,
                                        numFrames=test_numFrames,
                                        transform=transforms['val'],
                                        NDI=args.numDynamicImagesPerVideo,
                                        videoSegmentLength=args.videoSegmentLength,
                                        positionSegment=args.positionSegment,
                                        overlapping=args.overlapping,
                                        frameSkip=args.frameSkip,
                                        skipInitialFrames=0,
                                        segmentPreprocessing=args.segmentPreprocessing,
                                        batchSize=args.batchSize,
                                        shuffle=True,
                                        numWorkers=args.numWorkers)
            
            model, _ = initialize_model(model_name=args.modelType,
                                        num_classes=2,
                                        feature_extract=args.featureExtract,
                                        numDiPerVideos=args.numDynamicImagesPerVideo,
                                        joinType=args.joinType,
                                        use_pretrained=True)
            model.to(DEVICE)
            params_to_update = verifiParametersToTrain(model, args.featureExtract)
            optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
            criterion = nn.CrossEntropyLoss()
            experimentConfig = 'HOCKEY-Model-{},segmentLen-{},numDynIms-{},frameSkip-{},segmentPreprocessing-{},epochs-{},split_type-{},fold-{}'.format(args.modelType,
                                                                                                                                args.videoSegmentLength,
                                                                                                                                args.numDynamicImagesPerVideo,
                                                                                                                                args.frameSkip,
                                                                                                                                args.segmentPreprocessing,
                                                                                                                                args.numEpochs,
                                                                                                                                args.split_type,
                                                                                                                                fold)
            
            log_dir = os.path.join(constants.PATH_RESULTS, 'HOCKEY', 'tensorboard-runs', experimentConfig)
            writer = SummaryWriter(log_dir)
            print('Tensorboard logDir={}'.format(log_dir))
            tr = trainer.Trainer(model=model,
                            model_transfer= args.transferModel,
                            train_dataloader=train_dt_loader.getDataloader(),
                            val_dataloader=test_dt_loader.getDataloader(),
                            criterion=criterion,
                            optimizer=optimizer,
                            num_epochs=args.numEpochs,
                            checkpoint_path=os.path.join(constants.PATH_RESULTS, 'HOCKEY', 'checkpoints', experimentConfig),
                            lr_scheduler=None)
            
            policy = ResultPolicy()

            for epoch in range(1, args.numEpochs + 1):
                print("Fold {} ----- Epoch {}/{}".format(fold,epoch, args.numEpochs))
                # Train and evaluate
                epoch_loss_train, epoch_acc_train = tr.train_epoch(epoch)
                epoch_loss_val, epoch_acc_val = tr.val_epoch(epoch)
                exp_lr_scheduler.step()
                flac = policy.update(epoch_loss_train, epoch_acc_train, epoch_loss_val, epoch_acc_val, epoch)
                if args.saveCheckpoint:
                    tr.saveCheckpoint(epoch, flac)

                writer.add_scalar('training loss', epoch_loss_train, epoch)
                writer.add_scalar('validation loss', epoch_loss_val, epoch)
                writer.add_scalar('training Acc', epoch_acc_train, epoch)
                writer.add_scalar('validation Acc', epoch_acc_val, epoch)
            cv_test_accs.append(policy.getFinalTestAcc())
            cv_test_losses.append(policy.getFinalTestLoss())
            cv_final_epochs.append(policy.getFinalEpoch())
   
    print('CV Accuracies=', cv_test_accs)
    print('CV Losses=', cv_test_losses)
    print('CV Epochs=', cv_final_epochs)
    print('Test AVG Accuracy={}, Test AVG Loss={}'.format(np.average(cv_test_accs), np.average(cv_test_losses)))

__main__()