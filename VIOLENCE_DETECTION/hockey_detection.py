

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

from violenceDataset import *
from operator import itemgetter
import random
from hockey_transforms import *
# import UTIL.initializeDataset as initializeDataset
from UTIL.chooseModel import initialize_model
from UTIL.parameters import verifiParametersToTrain
import UTIL.trainer as trainer
import UTIL.tester as tester
from UTIL.kfolds import k_folds
from UTIL.util import save_csvfile_multicolumn, read_csvfile_threecolumns, read_file
from constants import DEVICE
from UTIL.resultsPolicy import ResultPolicy
import pandas as pd
# from FPS import FPSMeter
from torch.utils.tensorboard import SummaryWriter
import csv

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


    args = parser.parse_args()
    split_type = args.split_type
    input_size = 224
    shuffle = True
    path_violence = constants.PATH_HOCKEY_FRAMES_VIOLENCE
    path_non_violence = constants.PATH_HOCKEY_FRAMES_NON_VIOLENCE

    if not os.path.exists(os.path.join(constants.PATH_HOCKEY_README, 'all_data_labels_numFrames.csv')):
        datasetAll, labelsAll, numFramesAll = initializeDataset.createDataset(path_violence, path_non_violence, shuffle)  #shuffle
        all_data = zip(datasetAll, labelsAll, numFramesAll)
        save_csvfile_multicolumn(all_data, os.path.join(constants.PATH_HOCKEY_README, 'all_data_labels_numFrames.csv'))
    else:
        datasetAll, labelsAll, numFramesAll = read_csvfile_threecolumns(os.path.join(constants.PATH_HOCKEY_README, 'all_data_labels_numFrames.csv'))


    transforms = createTransforms(input_size)
    cv_test_accs = []
    cv_test_losses = []
    cv_final_epochs = []

    if split_type[:-2] == 'train-test':
        train_idx = read_file(os.path.join(constants.PATH_HOCKEY_README, 'fold_{}_train.txt'.format(int(split_type[len(split_type)-1]))))
        test_idx = read_file(os.path.join(constants.PATH_HOCKEY_README, 'fold_{}_test.txt'.format(int(split_type[len(split_type)-1]))))
        train_idx = list(map(int, train_idx))
        test_idx = list(map(int, test_idx))
        
        train_x = list(itemgetter(*train_idx)(datasetAll))
        train_y = list(itemgetter(*train_idx)(labelsAll))
        train_numFrames = list(itemgetter(*train_idx)(numFramesAll))
        test_x = list(itemgetter(*test_idx)(datasetAll))
        test_y = list(itemgetter(*test_idx)(labelsAll))
        test_numFrames = list(itemgetter(*test_idx)(numFramesAll))

        # initializeDataset.print_balance(train_y, test_y)

        # dataloaders_dict = initializeDataset.getDataLoaders(train_x, train_y, train_numFrames, test_x, test_y, test_numFrames,
        #                                         transforms, numDynamicImagesPerVideo, train_batch_size=batch_size, test_batch_size=1,
        #                                         train_num_workers=num_workers, test_num_workers=1, videoSegmentLength=videoSegmentLength,
        #                                         positionSegment=positionSegment, overlaping=overlaping, frame_skip=frame_skip)
        image_datasets = {
            "train": ViolenceDataset(dataset=train_x,
                                    labels=train_y,
                                    numFrames=train_numFrames,
                                    spatial_transform=transforms["train"],
                                    numDynamicImagesPerVideo=args.numDynamicImagesPerVideo,
                                    videoSegmentLength=args.videoSegmentLength,
                                    positionSegment=args.positionSegment,
                                    overlaping=args.overlapping,
                                    frame_skip=args.frameSkip),
            "val": ViolenceDataset(dataset=test_x,
                                    labels=test_y,
                                    numFrames=test_numFrames,
                                    spatial_transform=transforms["val"],
                                    numDynamicImagesPerVideo=args.numDynamicImagesPerVideo,
                                    videoSegmentLength=args.videoSegmentLength,
                                    positionSegment=args.positionSegment,
                                    overlaping=args.overlapping,
                                    frame_skip=args.frameSkip),
        }
        dataloaders_dict = {
            "train": torch.utils.data.DataLoader(image_datasets["train"], batch_size=args.batchSize, shuffle=shuffle, num_workers=args.numWorkers),
            "val": torch.utils.data.DataLoader( image_datasets["val"], batch_size=args.batchSize, shuffle=shuffle, num_workers=args.numWorkers)
        }
            
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
        # split_type = split_type+str(fold)
        experimentConfig = 'HOCKEY-Model-{},segmentLen-{},numDynIms-{},frameSkip-{},epochs-{},split_type-{}'.format(args.modelType,
                                                                                                                    args.videoSegmentLength,
                                                                                                                    args.numDynamicImagesPerVideo,
                                                                                                                    args.frameSkip,
                                                                                                                    args.numEpochs,
                                                                                                                    args.split_type)
        log_dir = os.path.join(constants.PATH_RESULTS, 'HOCKEY', 'tensorboard-runs', experimentConfig)
        writer = SummaryWriter(log_dir)
        print('Tensorboard logDir={}'.format(log_dir))
        tr = trainer.Trainer(model=model,
                            model_transfer= args.transferModel,
                            train_dataloader=dataloaders_dict['train'],
                            val_dataloader=dataloaders_dict['val'],
                            criterion=criterion,
                            optimizer=optimizer,
                            num_epochs=args.numEpochs,
                            checkpoint_path=os.path.join(constants.PATH_RESULTS,'HOCKEY','checkpoints',experimentConfig+ '.tar'))
        
        final_val_loss = 1000
        final_train_loss = 1000
        final_val_acc = 0
        best_epoch = 0
        policy = ResultPolicy()

        for epoch in range(1, args.numEpochs + 1):
            print("----- Epoch {}/{}".format(epoch, args.numEpochs))
            # Train and evaluate
            epoch_loss_train, epoch_acc_train = tr.train_epoch(epoch)
            epoch_loss_val, epoch_acc_val = tr.val_epoch(epoch)
            exp_lr_scheduler.step()

            flac = policy.update(epoch_loss_train, epoch_acc_train, epoch_loss_val, epoch_acc_val, epoch)
            # print(flac, type(flac))
            tr.saveCheckpoint(epoch, flac)

            writer.add_scalar('training loss', epoch_loss_train, epoch)
            writer.add_scalar('validation loss', epoch_loss_val, epoch)
            writer.add_scalar('training Acc', epoch_acc_train, epoch)
            writer.add_scalar('validation Acc', epoch_acc_val, epoch)

            # # if final_val_loss
            # if epoch_loss_train >= epoch_loss_val and epoch_acc_val > final_val_acc:
            #     final_val_acc = epoch_acc_val
            #     final_val_loss = epoch_loss_val
            #     final_train_loss = epoch_loss_train
            #     best_epoch = epoch
        cv_test_accs.append(policy.getFinalTestAcc())
        cv_test_losses.append(policy.getFinalTestLoss())
        cv_final_epochs.append(policy.getFinalEpoch())

        # print('Best Validation Accuracy={} in Epoch ({}) with val_loss ({}) <  train_loss ({})'.format(final_val_acc, best_epoch, final_val_loss, final_train_loss))
            
    elif split_type == 'cross-val':
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
            # initializeDataset.print_balance(train_y, test_y)

            image_datasets = {
            "train": ViolenceDataset(dataset=train_x,
                                    labels=train_y,
                                    numFrames=train_numFrames,
                                    spatial_transform=transforms["train"],
                                    numDynamicImagesPerVideo=args.numDynamicImagesPerVideo,
                                    videoSegmentLength=args.videoSegmentLength,
                                    positionSegment=args.positionSegment,
                                    overlaping=args.overlapping,
                                    frame_skip=args.frameSkip),
            "val": ViolenceDataset(dataset=test_x,
                                    labels=test_y,
                                    numFrames=test_numFrames,
                                    spatial_transform=transforms["val"],
                                    numDynamicImagesPerVideo=args.numDynamicImagesPerVideo,
                                    videoSegmentLength=args.videoSegmentLength,
                                    positionSegment=args.positionSegment,
                                    overlaping=args.overlapping,
                                    frame_skip=args.frameSkip),
            }
            dataloaders_dict = {
                "train": torch.utils.data.DataLoader(image_datasets["train"], batch_size=args.batchSize, shuffle=shuffle, num_workers=args.numWorkers),
                "val": torch.utils.data.DataLoader( image_datasets["val"], batch_size=args.batchSize, shuffle=shuffle, num_workers=args.numWorkers)
            }
            
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
            # split_type = split_type+str(fold)
            experimentConfig = 'HOCKEY-Model-{},segmentLen-{},numDynIms-{},frameSkip-{},epochs-{},split_type-{},fold-{}'.format(args.modelType,
                                                                                                                                args.videoSegmentLength,
                                                                                                                                args.numDynamicImagesPerVideo,
                                                                                                                                args.frameSkip,
                                                                                                                                args.numEpochs,
                                                                                                                                args.split_type,
                                                                                                                                fold)
            
            log_dir = os.path.join(constants.PATH_RESULTS, 'HOCKEY', 'tensorboard-runs', experimentConfig)
            writer = SummaryWriter(log_dir)
            print('Tensorboard logDir={}'.format(log_dir))
            tr = trainer.Trainer(model=model,
                            model_transfer= args.transferModel,
                            train_dataloader=dataloaders_dict['train'],
                            val_dataloader=dataloaders_dict['val'],
                            criterion=criterion,
                            optimizer=optimizer,
                            num_epochs=args.numEpochs,
                            checkpoint_path=os.path.join(constants.PATH_RESULTS,'HOCKEY','checkpoints',experimentConfig))
            
            policy = ResultPolicy()

            for epoch in range(1, args.numEpochs + 1):
                print("Fold {} ----- Epoch {}/{}".format(fold,epoch, args.numEpochs))
                # Train and evaluate
                epoch_loss_train, epoch_acc_train = tr.train_epoch(epoch)
                epoch_loss_val, epoch_acc_val = tr.val_epoch(epoch)
                exp_lr_scheduler.step()
                flac = policy.update(epoch_loss_train, epoch_acc_train, epoch_loss_val, epoch_acc_val, epoch)
                # print(flac, type(flac))
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
    
# __main_mask__()
__main__()