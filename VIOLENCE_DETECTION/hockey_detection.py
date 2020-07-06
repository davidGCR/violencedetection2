

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
from transforms import hockeyTransforms
from UTIL.chooseModel import initialize_model, initialize_FCNN
from UTIL.parameters import verifiParametersToTrain
import UTIL.trainer as trainer
import UTIL.tester as tester
from UTIL.kfolds import k_folds
from UTIL.util import  read_file, expConfig
from constants import DEVICE
# from UTIL.resultsPolicy import ResultPolicy
from UTIL.earlyStop import EarlyStopping
import pandas as pd
# from FPS import FPSMeter
from torch.utils.tensorboard import SummaryWriter
from datasetsMemoryLoader import hockeyLoadData, hockeyTrainTestSplit
from dataloader import MyDataloader
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
    parser.add_argument("--skipInitialFrames", type=int, default=0)
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

    if args.split_type[:-2] == 'fully-conv':
        
        train_x, train_y, train_numFrames, test_x, test_y, test_numFrames = hockeyTrainTestSplit(args.split_type, datasetAll, labelsAll, numFramesAll)
        default_args = {
            'X': train_x,
            'y': train_y,
            'numFrames': train_numFrames,
            'transform': transforms['train'],
            'NDI': 1,
            'videoSegmentLength': 30,
            'positionSegment': 'begin',
            'overlapping': 0,
            'frameSkip': 0,
            'skipInitialFrames': 0,
            'batchSize': 8,
            'shuffle': False,
            'numWorkers': 4,
            'pptype': None,
            'modelType': 'alexnetv2'
        }
        train_dt_loader = MyDataloader(default_args)
        default_args['X'] = test_x
        default_args['y'] = test_y
        default_args['numFrames'] = test_numFrames
        default_args['transform'] = transforms['test']
        test_dt_loader = MyDataloader(default_args)
        
        model, _ = initialize_model(model_name=args.modelType,
                                    num_classes=2,
                                    feature_extract=args.featureExtract,
                                    numDiPerVideos=args.numDynamicImagesPerVideo,
                                    joinType=args.joinType,
                                    use_pretrained=True)

        if args.transferModel is not None:
            if DEVICE == 'cuda:0':
                model.load_state_dict(torch.load(args.transferModel), strict=False)
            else:
                model.load_state_dict(torch.load(args.transferModel, map_location=DEVICE))
        
        model = initialize_FCNN(model_name=args.modelType, original_model=model)
        print(model)
        model.eval()
        dataloaders = [train_dt_loader, test_dt_loader]
        outs = []
        labels = []
        for i, dt_loader in enumerate(dataloaders):
            for data in tqdm(dt_loader.dataloader):
                inputs, y, _, _ = data
                inputs = inputs.to(DEVICE)
                y = y.to(DEVICE)
                
                with torch.no_grad():
                    outputs = model(inputs)
                    outs.append(outputs)
                    labels.append(y)
            print('outs_loader({})=shape {}, type {}'.format(i+1,len(outs), type(outs)))
        outs = torch.stack(outs, dim=0)
        it, bacth, C, H, W = outs.size()
        outs = outs.view(it * bacth, C, H, W)
        outs = outs.permute(0, 2, 3, 1)
        # print('outs=', outs.size(), type(outs))
        n, H, W, C = outs.size()
        outs = outs.contiguous()
        outs = outs.view(n * H * W, C)
        # print('outs=', outs.size(), type(outs))
        outs = outs.numpy()
        # print('labels list=',len(labels))
        labels = torch.cat(labels, dim=0)
        labels = labels.numpy()
        # print('labels=', labels.shape, type(labels))
        # print(labels)
        # print('conv5_train_test({})=shape {}, type {}'.format(i+1,outs.shape, type(outs)))
        sio.savemat(file_name=os.path.join('/Users/davidchoqueluqueroman/Google Drive/ITQData','conv5_train_test-finetuned=No.mat'),mdict={'alexnetv2_cvv':outs, 'labels':labels})
        

    elif args.split_type[:-2] == 'train-test':
        train_x, train_y, train_numFrames, test_x, test_y, test_numFrames = hockeyTrainTestSplit(args.split_type, datasetAll, labelsAll, numFramesAll)
        default_args = {
            'X': train_x,
            'y': train_y,
            'numFrames': train_numFrames,
            'transform': transforms['train'],
            'NDI': args.numDynamicImagesPerVideo,
            'videoSegmentLength': args.videoSegmentLength,
            'positionSegment': args.positionSegment,
            'overlapping': args.overlapping,
            'frameSkip': args.frameSkip,
            'skipInitialFrames': 0,
            'batchSize': args.batchSize,
            'shuffle': True,
            'numWorkers': args.numWorkers,
            'pptype': None,
            'modelType': args.modelType
        }
        train_dt_loader = MyDataloader(default_args)
        default_args['X'] = test_x
        default_args['y'] = test_y
        default_args['numFrames'] = test_numFrames
        default_args['transform'] = transforms['val']
        test_dt_loader = MyDataloader(default_args)
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
        # print('Tensorboard logDir={}'.format(log_dir))
        tr = trainer.Trainer(model=model,
                            model_transfer= args.transferModel,
                            train_dataloader=train_dt_loader.getDataloader(),
                            val_dataloader=test_dt_loader.getDataloader(),
                            criterion=criterion,
                            optimizer=optimizer,
                            num_epochs=args.numEpochs,
                            checkpoint_path=None,
                            lr_scheduler=None)
        # policy = ResultPolicy()
        for epoch in range(1, args.numEpochs + 1):
            print("----- Epoch {}/{}".format(epoch, args.numEpochs))
            epoch_loss_train, epoch_acc_train = tr.train_epoch(epoch)
            epoch_loss_val, epoch_acc_val = tr.val_epoch(epoch)
            exp_lr_scheduler.step()
            # flac = policy.update(epoch_loss_train, epoch_acc_train, epoch_loss_val, epoch_acc_val, epoch)
            # if args.saveCheckpoint:
            #     tr.saveCheckpoint(epoch, flac, epoch_acc_val, epoch_loss_val)
            writer.add_scalar('training loss', epoch_loss_train, epoch)
            writer.add_scalar('validation loss', epoch_loss_val, epoch)
            writer.add_scalar('training Acc', epoch_acc_train, epoch)
            writer.add_scalar('validation Acc', epoch_acc_val, epoch)
        # cv_test_accs.append(policy.getFinalTestAcc())
        # cv_test_losses.append(policy.getFinalTestLoss())
        # cv_final_epochs.append(policy.getFinalEpoch())
            
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

            default_args = {
                'X': train_x,
                'y': train_y,
                'numFrames': train_numFrames,
                'transform': transforms['train'],
                'NDI': args.numDynamicImagesPerVideo,
                'videoSegmentLength': args.videoSegmentLength,
                'positionSegment': args.positionSegment,
                'overlapping': args.overlapping,
                'frameSkip': args.frameSkip,
                'skipInitialFrames': 0,
                'batchSize': args.batchSize,
                'shuffle': True,
                'numWorkers': args.numWorkers,
                'pptype': None,
                'modelType': args.modelType
            }
            train_dt_loader = MyDataloader(default_args)
            default_args['X'] = test_x
            default_args['y'] = test_y
            default_args['numFrames'] = test_numFrames
            default_args['transform'] = transforms['val']
            test_dt_loader = MyDataloader(default_args)
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
            config = expConfig(dataset='HOCKEY',
                                    modelType=args.modelType,
                                    featureExtract=args.featureExtract,
                                    numDynamicImages=args.numDynamicImagesPerVideo,
                                    segmentLength=args.videoSegmentLength,
                                    frameSkip=args.frameSkip,
                                    skipInitialFrames=args.skipInitialFrames,
                                    overlap=args.overlapping,
                                    joinType=args.joinType)
            log_dir = os.path.join(constants.PATH_RESULTS, 'HOCKEY', 'tensorboard-runs', experimentConfig)
            writer = SummaryWriter(log_dir)
            # print('Tensorboard logDir={}'.format(log_dir))
            tr = trainer.Trainer(model=model,
                            model_transfer= args.transferModel,
                            train_dataloader=train_dt_loader.dataloader,
                            val_dataloader=test_dt_loader.dataloader,
                            criterion=criterion,
                            optimizer=optimizer,
                            num_epochs=args.numEpochs,
                            checkpoint_path=None,
                            lr_scheduler=None)
            
            if args.saveCheckpoint:
                path = os.path.join(constants.PATH_RESULTS, 'HOCKEY', 'checkpoints', experimentConfig)
            else:
                path = None
            early_stopping = EarlyStopping(patience=5, verbose=True, path=path, model_config=config)
            for epoch in range(1, args.numEpochs + 1):
                print("Fold {} ----- Epoch {}/{}".format(fold,epoch, args.numEpochs))
                # Train and evaluate
                epoch_loss_train, epoch_acc_train = tr.train_epoch(epoch)
                exp_lr_scheduler.step()
                # flac, stop = policy.update(epoch_loss_train, epoch_acc_train, epoch_loss_val, epoch_acc_val, epoch)
                epoch_loss_val, epoch_acc_val = tr.val_epoch(epoch)
                early_stopping(epoch_loss_val, epoch_acc_val, epoch_loss_train, epoch, fold, tr.getModel())

                writer.add_scalar('training loss', epoch_loss_train, epoch)
                writer.add_scalar('validation loss', epoch_loss_val, epoch)
                writer.add_scalar('training Acc', epoch_acc_train, epoch)
                writer.add_scalar('validation Acc', epoch_acc_val, epoch)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
                
            cv_test_accs.append(early_stopping.best_acc)
            cv_test_losses.append(early_stopping.val_loss_min)
            cv_final_epochs.append(early_stopping.best_epoch)
   
    print('CV Accuracies=', cv_test_accs)
    print('CV Losses=', cv_test_losses)
    print('CV Epochs=', cv_final_epochs)
    print('Test AVG Accuracy={}, Test AVG Loss={}'.format(np.average(cv_test_accs), np.average(cv_test_losses)))
    print("Accuracy: %0.3f (+/- %0.3f), Losses: %0.3f" % (np.array(cv_test_accs).mean(), np.array(cv_test_accs).std() * 2, np.array(cv_test_losses).mean()))

__main__()