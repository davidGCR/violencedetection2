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

from UTIL.chooseModel import initialize_model, initialize_FCNN
from UTIL.parameters import verifiParametersToTrain
import UTIL.trainer as trainer
import UTIL.tester as tester
# from UTIL.kfolds import k_folds
# from UTIL.util import  read_file, expConfig
from constants import DEVICE
# from UTIL.resultsPolicy import ResultPolicy
from UTIL.earlyStop import EarlyStopping
import pandas as pd
# from FPS import FPSMeter
from torch.utils.tensorboard import SummaryWriter
from datasetsMemoryLoader import customize_kfold
# from dataloader import MyDataloader
from VIOLENCE_DETECTION.violenceDataset import ViolenceDataset
from VIOLENCE_DETECTION.rgbDataset import RGBDataset
import csv
from operator import itemgetter
from VIOLENCE_DETECTION.UTIL2 import base_dataset, load_model, transforms_dataset, plot_example
from collections import Counter
from sklearn.model_selection import train_test_split
import wandb

def train_model(model, dataloaders, criterion, optimizer, num_epochs, patience, fold, path, model_config, phases, metric_to_track=None):
    since = time.time()
    val_acc_history = []
    # best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    early_stopping = EarlyStopping(patience=patience, verbose=True, path=path, model_config=model_config)

    for epoch in range(1,num_epochs+1,1):
        print('Fold ({})--Epoch {}/{}'.format(fold, epoch, num_epochs+1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        last_train_loss = np.inf
        last_train_acc = np.inf
        for phase in phases:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for data in tqdm(dataloaders[phase]):
                # (inp, vid_name, dynamicImages, bboxes, rgb_central_frames) = inp
                (inputs, idx, dynamicImages, one_box, rgb_central_frames), labels = data
                # (inputs, idx, dynamicImages, one_box) = X

                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                # gt_bboxes = gt_bboxes.to(DEVICE)
                one_box = one_box.to(DEVICE)
                # zero the parameter gradients
                optimizer.zero_grad()
                batch_size = inputs.size()[0]
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # outputs = model(inputs, one_box) #for roi_pool
                    outputs = model((inputs, idx, dynamicImages, one_box))
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
                wandb.log({"Test Accuracy": epoch_loss, "Test Loss": epoch_acc})
            else:
                wandb.log({"Train Accuracy": epoch_loss, "Train Loss": epoch_acc})


            if phase == 'val':
                early_stopping(epoch_loss, epoch_acc, last_train_loss, epoch, fold, model)
            else:
                last_train_loss = epoch_loss
                last_train_acc = epoch_acc
                if metric_to_track == 'train-loss':
                    early_stopping(epoch_loss, epoch_acc, last_train_loss, epoch, fold, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return model, early_stopping.best_acc, early_stopping.val_loss_min, early_stopping.best_epoch

def test_model(model, dataloader):
    model.eval()   # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0
    # Iterate over data.
    for data in tqdm(dataloader):
        inputs, labels, v_names, _, _ = data
        # print('inputs=', inputs.size(), type(inputs))
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        batch_size = inputs.size()[0]
        # forward
        with torch.set_grad_enabled(False):
            # Get model outputs and calculate loss
            outputs = model(inputs)
            # loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

        # statistics
        # running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)


    # epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)

    print('{} Loss: {:.4f} Acc: {:.4f}'.format('Test', 0, epoch_acc))

def openSet_experiments(mode, args):
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--mode",type=str)
    if mode == 'openSet-train':
        # datasets = args.splitType[1:]
        # print(datasets)
        datasetAll, labelsAll, numFramesAll, transforms = [], [], [], []
        for dt in args.dataset:
            print(dt)
            x, y, num, tr = base_dataset(dt)
            datasetAll += x
            labelsAll += y
            numFramesAll += num
            transforms.append(tr)
        combined = list(zip(datasetAll, labelsAll, numFramesAll))
        random.shuffle(combined)
        datasetAll[:], labelsAll[:], numFramesAll[:] = zip(*combined)

        train_dataset = ViolenceDataset(dataset=datasetAll,
                                        labels=labelsAll,
                                        numFrames=numFramesAll,
                                        spatial_transform=transforms[0]['train'],
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
        dataloaders = {'train': train_dataloader}

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
        fold = 0
        checkpoint_path=None
        config = None
        if args.saveCheckpoint:
            config = {
                'dataset':args.dataset,
                'model': args.modelType,
                'numEpochs': args.numEpochs,
                'freezeConvLayers': args.freezeConvLayers,
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
            ss = ""
            for (key, val) in config.items():
                if key != 'log_dir':
                    ss = ss + "_{!s}={!r}".format(key, val)
            ss = ss.replace("\'", "")
            # print(ss)
            checkpoint_path = os.path.join(constants.PATH_RESULTS, 'OPENSET', 'checkpoints', 'DYN_Stream-{}-fold={}'.format(ss,fold))

        phases = ['train']
        model, best_acc, val_loss_min, best_epoch = train_model(model,
                                                            dataloaders,
                                                            criterion,
                                                            optimizer,
                                                            num_epochs=args.numEpochs,
                                                            patience=args.patience,
                                                            fold=fold,
                                                            path=checkpoint_path,
                                                            model_config=config,
                                                            phases = phases,
                                                            metric_to_track='train-loss')

    elif mode == 'openSet-test':

        ## Load model
        checkpoint = torch.load(args.modelPath, map_location=DEVICE)
        model = checkpoint['model_config']['model']
        numDynamicImages = checkpoint['model_config']['numDynamicImages']
        joinType = checkpoint['model_config']['joinType']
        freezeConvLayers = checkpoint['model_config']['freezeConvLayers']
        videoSegmentLength = checkpoint['model_config']['segmentLength']
        overlapping = checkpoint['model_config']['overlap']
        frameSkip = checkpoint['model_config']['frameSkip']
        skipInitialFrames = checkpoint['model_config']['skipInitialFrames']
        useKeyframes = checkpoint['model_config']['useKeyframes']
        windowLen = checkpoint['model_config']['windowLen']

        model_, _ = initialize_model(model_name=model,
                                        num_classes=2,
                                        freezeConvLayers=freezeConvLayers,
                                        numDiPerVideos=numDynamicImages,
                                        joinType=joinType,
                                        use_pretrained=True)

        model_.to(DEVICE)
        # print(model_)
        if DEVICE == 'cuda:0':
            model_.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model_.load_state_dict(checkpoint['model_state_dict'])


        datasetAll, labelsAll, numFramesAll, transforms = base_dataset(args.testDataset)
        test_dataset = ViolenceDataset(dataset=datasetAll,
                                        labels=labelsAll,
                                        numFrames=numFramesAll,
                                        spatial_transform=transforms['val'],
                                        numDynamicImagesPerVideo=numDynamicImages,
                                        videoSegmentLength=videoSegmentLength,
                                        positionSegment=None,
                                        overlaping=overlapping,
                                        frame_skip=frameSkip,
                                        skipInitialFrames=skipInitialFrames,
                                        ppType=None,
                                        useKeyframes=useKeyframes,
                                        windowLen=windowLen)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=True, num_workers=4)

        test_model(model_, test_dataloader)



def args_2_checkpoint_path(args, fold=0):
    config = {
                'dataset':args.dataset,
                'model': args.modelType,
                'numEpochs': args.numEpochs,
                'freezeConvLayers': args.freezeConvLayers,
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
    # datasets=''
    # for dt in args.dataset:
    #     datasets += '-'+dt
    checkpoint_path = os.path.join(constants.PATH_RESULTS, args.dataset[0].upper(), 'checkpoints', 'TemporalStream_Best_model-{}-fold={}.pt'.format(ss, fold))
    return checkpoint_path

def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lib", type=str)
    parser.add_argument("--modelType", type=str, default="alexnet", help="model")
    parser.add_argument("--inputSize", type=int)
    parser.add_argument("--useValSplit", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--dataset", nargs='+', type=str)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--numEpochs",type=int,default=30)
    parser.add_argument("--batchSize",type=int,default=64)
    parser.add_argument("--freezeConvLayers",type=lambda x: (str(x).lower() == 'true'), default=False, help="to fine tunning")
    parser.add_argument("--numDynamicImagesPerVideo", type=int )
    parser.add_argument("--joinType", type=str)
    parser.add_argument("--foldsNumber", type=int, default=5)
    parser.add_argument("--numWorkers", type=int, default=4)
    parser.add_argument("--videoSegmentLength", type=int)
    parser.add_argument("--positionSegment", type=str)
    parser.add_argument("--splitType", type=str)
    parser.add_argument("--overlapping", type=float)
    parser.add_argument("--frameSkip", type=int, default=0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--skipInitialFrames", type=int, default=0)
    parser.add_argument("--saveCheckpoint", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--useKeyframes", type=str, default=None)
    parser.add_argument("--windowLen", type=int, default=0)
    parser.add_argument("--modelPath", type=str, default=None)
    parser.add_argument("--testDataset",type=str, default=None)
    parser.add_argument("--pretrained", type=lambda x: (str(x).lower() == 'true'), default=False, help="to fine tunning")
    parser.add_argument("--lossCoefGlobal", type=float, default=0.8)
    parser.add_argument("--lossCoefLocal", type=float, default=0.1)
    parser.add_argument("--lossCoefFusion", type=float, default=0.1)
    parser.add_argument("--trainOneModel", type=str, default="")
    parser.add_argument("--optimizer", type=str, default="adam")

    args = parser.parse_args()
    return args


def __pytorch__(args):
    # args = build_args()

    wandb.login()

    if args.splitType == 'cam':
         __weakly_localization__()
         return 0

    if args.splitType == 'openSet-train' or args.splitType == 'openSet-test':
        openSet_experiments(mode=args.splitType, args=args)
        return 0

    input_size = 224
    shuffle = True
    if args.dataset[0] == 'rwf-2000':
        datasetAll, labelsAll = [], []
    else:
        datasetAll, labelsAll, numFramesAll, transforms = base_dataset(args.dataset[0], input_size=args.inputSize)
        print('====> Loaded all dataset? X:{}, y:{}, numFrames:{}'.format(len(datasetAll), len(labelsAll), len(numFramesAll)))
    cv_test_accs = []
    cv_test_losses = []
    cv_final_epochs = []
    # patience = 5
    folds_number = 5
    fold = 0
    checkpoint_path = None
    config = None
    print(args.dataset)

    # for train_idx, test_idx in customize_kfold(n_splits=folds_number, dataset=args.dataset[0], X_len=len(datasetAll), shuffle=shuffle):
    for train_idx, test_idx in customize_kfold(n_splits=folds_number, dataset=args.dataset[0], X=datasetAll, y=labelsAll, shuffle=shuffle):
        fold = fold + 1
        wandb_run = wandb.init(project="pytorch-violencedetection2")
        wandb_run.config.update(vars(args))

        print("**************** Fold:{}/{} ".format(fold, folds_number))
        if args.dataset[0] == 'rwf-2000':
            train_x, train_y, train_numFrames, test_x, test_y, test_numFrames, transforms = base_dataset(args.dataset[0], input_size=args.inputSize)
            print(len(train_x), len(train_y), len(train_numFrames), len(test_x), len(test_y), len(test_numFrames))
        else:
            train_x, train_y, test_x, test_y = None, None, None, None
            train_x = list(itemgetter(*train_idx)(datasetAll))
            train_y = list(itemgetter(*train_idx)(labelsAll))
            train_numFrames = list(itemgetter(*train_idx)(numFramesAll))
            test_x = list(itemgetter(*test_idx)(datasetAll))
            test_y = list(itemgetter(*test_idx)(labelsAll))
            test_numFrames = list(itemgetter(*test_idx)(numFramesAll))


        test_dataset = ViolenceDataset(videos=test_x,
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
                                        windowLen=args.windowLen,
                                        dataset=args.dataset[0])

        print('Label distribution:')
        print('Train=', Counter(train_y))
        # print('Val=', Counter(val_y))
        print('Test=', Counter(test_y))
        if args.useValSplit:
            train_x, val_x, train_numFrames, val_numFrames, train_y, val_y = train_test_split(train_x, train_numFrames, train_y, test_size=0.2, stratify=train_y, random_state=1)
            #Label distribution
            # print('Label distribution:')
            # print('Train=', Counter(train_y))
            print('Val=', Counter(val_y))
            # print('Test=', Counter(test_y))

            val_dataset = ViolenceDataset(videos=val_x,
                                            labels=val_y,
                                            numFrames=val_numFrames,
                                            spatial_transform=transforms['val'],
                                            numDynamicImagesPerVideo=args.numDynamicImagesPerVideo,
                                            videoSegmentLength=args.videoSegmentLength,
                                            positionSegment=args.positionSegment,
                                            overlaping=args.overlapping,
                                            frame_skip=args.frameSkip,
                                            skipInitialFrames=args.skipInitialFrames,
                                            ppType=None,
                                            useKeyframes=args.useKeyframes,
                                            windowLen=args.windowLen,
                                            dataset=args.dataset[0])

        train_dataset = ViolenceDataset(videos=train_x,
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
                                        windowLen=args.windowLen,
                                        dataset=args.dataset[0])

        test_dataset = ViolenceDataset(videos=test_x,
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
                                        windowLen=args.windowLen,
                                        dataset=args.dataset[0])


        if not args.useValSplit:
            val_dataset = test_dataset

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=True, num_workers=args.numWorkers)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchSize, shuffle=True, num_workers=args.numWorkers)

        dataloaders = {'train': train_dataloader, 'val': val_dataloader}

        # if args.pretrained is not None:
        #     model = load_model(args.pretrained)
        # else:
        #     model = initialize_model(model_name=args.modelType,
        #                                 num_classes=2,
        #                                 freezeConvLayers=args.freezeConvLayers,
        #                                 numDiPerVideos=args.numDynamicImagesPerVideo,
        #                                 joinType=args.joinType,
        #                                 pretrained=args.pretrained)
        model = initialize_model(model_name=args.modelType,
                                    num_classes=2,
                                    freezeConvLayers=args.freezeConvLayers,
                                    numDiPerVideos=args.numDynamicImagesPerVideo,
                                    joinType=args.joinType,
                                    pretrained=args.pretrained)
        model.to(DEVICE)

        if args.modelType == 'c3d_v2':
            from MODELS.c3d_v2 import get_1x_lr_params, get_10x_lr_params
            params_to_update = [{'params': get_1x_lr_params(model), 'lr': args.lr},
                            {'params': get_10x_lr_params(model), 'lr': args.lr * 10}]
            optimizer = optim.SGD(params_to_update, lr=args.lr, momentum=0.9, weight_decay=5e-4)
            
            # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        else:
            params_to_update = verifiParametersToTrain(model, args.freezeConvLayers, printLayers=True)
            if args.optimizer == 'adam':
                optimizer = optim.Adam(params_to_update, lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
            else:
                optimizer = optim.SGD(params_to_update, lr=args.lr, momentum=0.9)
            

        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        criterion = nn.CrossEntropyLoss()
        wandb.watch(model, log="all")

        # log_dir = os.path.join(constants.PATH_RESULTS, args.dataset.upper(), 'tensorboard-runs', experimentConfig_str)
        # writer = SummaryWriter(log_dir)
        # if args.saveCheckpoint:
        #     checkpoint_path = args_2_checkpoint_path(args, fold=fold)

        phases = ['train', 'val']
        model, best_acc, val_loss_min, best_epoch = train_model(model,
                                                            dataloaders,
                                                            criterion,
                                                            optimizer,
                                                            num_epochs=args.numEpochs,
                                                            patience=args.patience,
                                                            fold=fold,
                                                            path=checkpoint_path,
                                                            phases = phases,
                                                            model_config=config)
        cv_test_accs.append(best_acc.item())
        cv_test_losses.append(val_loss_min)
        cv_final_epochs.append(best_epoch)


    print('CV Accuracies=', cv_test_accs)
    print('CV Losses=', cv_test_losses)
    print('CV Epochs=', cv_final_epochs)
    print('Test AVG Accuracy={}, Test AVG Loss={}'.format(np.average(cv_test_accs), np.average(cv_test_losses)))
    print("Accuracy: %0.3f (+/- %0.3f), Losses: %0.3f" % (np.array(cv_test_accs).mean(), np.array(cv_test_accs).std() * 2, np.array(cv_test_losses).mean()))

from skorch import NeuralNetClassifier
from skorch.helper import predefined_split, SliceDataset
from skorch.callbacks import LRScheduler, Freezer, TensorBoard, Checkpoint, WandbLogger
from sklearn.metrics import accuracy_score
import wandb


def skorch_a(args):

    shuffle = True
    if args.dataset[0] == 'rwf-2000':
        datasetAll, labelsAll = [], []
    else:
        datasetAll, labelsAll, numFramesAll, transforms = base_dataset(args.dataset[0], input_size=args.inputSize)
    cv_test_accs = []
    cv_test_losses = []
    cv_final_epochs = []
    # patience = 5
    folds_number = 5
    fold = 0
    checkpoint_path = None
    config = None
    print(args.dataset)

    wandb_run = wandb.init('Violence-Detection-2')

    for train_idx, test_idx in customize_kfold(n_splits=folds_number, dataset=args.dataset[0], X=datasetAll, y=labelsAll, shuffle=shuffle):
        fold = fold + 1
        print("**************** Fold:{}/{} ".format(fold, folds_number))
        if args.dataset[0] == 'rwf-2000':
            train_x, train_y, train_numFrames, test_x, test_y, test_numFrames, transforms = base_dataset(args.dataset[0], input_size=args.inputSize)
            # print(len(train_x), len(train_y), len(train_numFrames), len(test_x), len(test_y), len(test_numFrames))
        else:
            train_x, train_y, test_x, test_y = None, None, None, None
            train_x = list(itemgetter(*train_idx)(datasetAll))
            train_y = list(itemgetter(*train_idx)(labelsAll))
            train_numFrames = list(itemgetter(*train_idx)(numFramesAll))
            test_x = list(itemgetter(*test_idx)(datasetAll))
            test_y = list(itemgetter(*test_idx)(labelsAll))
            test_numFrames = list(itemgetter(*test_idx)(numFramesAll))

        print('Label distribution:')
        print('Train=', Counter(train_y))
        # print('Val=', Counter(val_y))
        print('Test=', Counter(test_y))
        if args.useValSplit:
            train_x, val_x, train_numFrames, val_numFrames, train_y, val_y = train_test_split(train_x, train_numFrames, train_y, test_size=0.2, stratify=train_y, random_state=1)
            #Label distribution
            # print('Label distribution:')
            # print('Train=', Counter(train_y))
            print('Val=', Counter(val_y))
            # print('Test=', Counter(test_y))

            val_dataset = ViolenceDataset(videos=val_x,
                                            labels=val_y,
                                            numFrames=val_numFrames,
                                            spatial_transform=transforms['val'],
                                            numDynamicImagesPerVideo=args.numDynamicImagesPerVideo,
                                            videoSegmentLength=args.videoSegmentLength,
                                            positionSegment=args.positionSegment,
                                            overlaping=args.overlapping,
                                            frame_skip=args.frameSkip,
                                            skipInitialFrames=args.skipInitialFrames,
                                            ppType=None,
                                            useKeyframes=args.useKeyframes,
                                            windowLen=args.windowLen,
                                            dataset=args.dataset[0])

        train_dataset = ViolenceDataset(videos=train_x,
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
                                        windowLen=args.windowLen,
                                        dataset=args.dataset[0])

        test_dataset = ViolenceDataset(videos=test_x,
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
                                        windowLen=args.windowLen,
                                        dataset=args.dataset[0])

        if not args.useValSplit:
            val_dataset = test_dataset
        # from MODELS.ViolenceModels import ResNet
        # PretrainedModel = ResNet(num_classes=2,
        #                         numDiPerVideos=args.numDynamicImagesPerVideo,
        #                         model_name=args.modelType,
        #                         joinType=args.joinType,
        #                         freezeConvLayers=args.freezeConvLayers)

        PretrainedModel = initialize_model(model_name=args.modelType,
                                        num_classes=2,
                                        freezeConvLayers=args.freezeConvLayers,
                                        numDiPerVideos=args.numDynamicImagesPerVideo,
                                        joinType=args.joinType,
                                        pretrained=args.pretrained)

        lrscheduler = LRScheduler(policy='StepLR', step_size=7, gamma=0.1)


        freezer = Freezer(lambda x: not x.startswith('fc8'))


        # writer = SummaryWriter()
        # ts = TensorBoard(writer)
        # wandb_run = wandb.init('Violence-Detection-2')
        # wandb.init(group='fold-'.format(fold))
        wandb_run.config.update(vars(args))

        if args.freezeConvLayers:
            callbacks = [lrscheduler, freezer,  WandbLogger(wandb_run)]
        else:
            callbacks = [lrscheduler,  WandbLogger(wandb_run)]

        if args.saveCheckpoint:
            #checkpoint_path = args_2_checkpoint_path(args, fold=fold)
            #cp = Checkpoint(f_params=checkpoint_path, monitor='valid_acc_best')
            cp = Checkpoint(dirname='rwf-exp1', monitor='valid_acc_best')
            callbacks.append(cp)
            #print('checkpoint_path: ', checkpoint_path)

        optimizer=optim.SGD

        print('Running in: ', DEVICE)
        if DEVICE=='cpu':
            net = NeuralNetClassifier(
                    PretrainedModel,
                    criterion=nn.CrossEntropyLoss,
                    lr=args.lr,
                    batch_size=args.batchSize,
                    max_epochs=args.numEpochs,
                    optimizer=optimizer,
                    optimizer__momentum=0.9,
                    iterator_train__shuffle=True,
                    iterator_train__num_workers=4,
                    iterator_valid__shuffle=True,
                    iterator_valid__num_workers=args.numWorkers,
                    train_split=predefined_split(val_dataset),
                    callbacks=callbacks
                )
        else:
            net = NeuralNetClassifier(
                    PretrainedModel,
                    criterion=nn.CrossEntropyLoss,
                    lr=args.lr,
                    batch_size=args.batchSize,
                    max_epochs=args.numEpochs,
                    optimizer=optimizer,
                    optimizer__momentum=0.9,
                    iterator_train__shuffle=True,
                    iterator_train__num_workers=4,
                    iterator_valid__shuffle=True,
                    iterator_valid__num_workers=args.numWorkers,
                    train_split=predefined_split(val_dataset),
                    callbacks=callbacks,
                    device=DEVICE
                )
        net.fit(train_dataset, y=None);

        ## inferenceMode
        if args.saveCheckpoint:
            net = NeuralNetClassifier(
                    PretrainedModel,
                    criterion=nn.CrossEntropyLoss,
                    lr=args.lr,
                    batch_size=args.batchSize,
                    max_epochs=args.numEpochs,
                    optimizer=optimizer,
                    optimizer__momentum=0.9,
                    iterator_train__shuffle=True,
                    iterator_train__num_workers=4,
                    iterator_valid__shuffle=True,
                    iterator_valid__num_workers=args.numWorkers,
                    train_split=predefined_split(val_dataset),
                    callbacks=callbacks,
                    device=DEVICE
                )

            net.initialize()
            net.load_params(checkpoint=cp)

        y_pred = net.predict(test_dataset)
        X_test_s = SliceDataset(test_dataset)
        y_test_s = SliceDataset(test_dataset, idx=1)
        acc = accuracy_score(y_test_s, y_pred)
        cv_test_accs.append(acc)
        print('Accuracy={}'.format(acc))

        # error_mask = y_pred != y_test_s
        # print(error_mask)


        # plot_example(X_test_s[error_mask], y_pred[error_mask], test_x)

        # plot_example(X_test_s, y_test_s, test_x)

        # from sklearn.model_selection import GridSearchCV
        # params = {
        #     'lr': [0.03, 0.02, 0.01, 0.001],
        #     'max_epochs': [25, 35],
        # }
        # gs = GridSearchCV(net, params, refit=False, cv=3, scoring='accuracy',n_jobs=1, verbose=1)

        # from skorch.helper import SliceDataset

        # X_sl = SliceDataset(train_dataset, idx=0)  # idx=0 is the default
        # print(X_sl.shape)

        # # y_sl = np.array([y for x, y in iter(train_dataset)])
        # y_sl = SliceDataset(train_dataset, idx=1)
        # print(y_sl.shape)

        # try:
        #   gs.fit(X_sl, y_sl)
        # except Exception as e:
        #   print(e)

    print('CV Accuracies=', cv_test_accs)
    print('Test AVG Accuracy={}'.format(np.average(cv_test_accs)))
    print("Accuracy: %0.3f (+/- %0.3f)" % (np.array(cv_test_accs).mean(), np.array(cv_test_accs).std() * 2))


def __my_main__():
    args = build_args()
    if args.lib == 'pytorch':
        __pytorch__(args)
    else:
        skorch_a(args)

if __name__ == "__main__":
    # skorch_a()
    __my_main__()
    # __weakly_localization__()
