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
                inputs, labels, v_names, _, _ = data
                # print(v_names)
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
    

def base_dataset(dataset, mean=None, std=None):
    if dataset == 'ucfcrime2local':
        datasetAll, labelsAll, numFramesAll = crime2localLoadData(min_frames=40)
        transforms = ucf2CrimeTransforms(224, mean=mean, std=std)
    elif dataset == 'vif':
        # print('hereeeeeeeeee')
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

# def openSet_experiments_train():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--trainDataset",type=str)
#     args = parser.parse_args()

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

def load_model(modelPath):
    checkpoint = torch.load(modelPath, map_location=DEVICE)
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

    return model_

def __weakly_localization__():
    from VIOLENCE_DETECTION.CAM import compute_CAM
    from VIOLENCE_DETECTION.datasetsMemoryLoader import load_fold_data
    from VIDEO_REPRESENTATION.dynamicImage import getDynamicImage
    import cv2

    ## Load model
    modelPath = os.path.join(
        constants.PATH_RESULTS,
        'HOCKEY/checkpoints',
        'DYN_Stream-_dataset=[hockey]_model=resnet50_numEpochs=25_freezeConvLayers=True_numDynamicImages=1_segmentLength=30_frameSkip=0_skipInitialFrames=0_overlap=0.0_joinType=maxTempPool_useKeyframes=None_windowLen=0-fold=1.pt'
        )
    dataset='hockey'

    checkpoint = torch.load(modelPath, map_location=DEVICE)
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


    datasetAll, labelsAll, numFramesAll, transforms = base_dataset(dataset)
    train_idx, test_idx = load_fold_data(dataset, fold=1)
    train_x = list(itemgetter(*train_idx)(datasetAll))
    train_y = list(itemgetter(*train_idx)(labelsAll))
    train_numFrames = list(itemgetter(*train_idx)(numFramesAll))
    test_x = list(itemgetter(*test_idx)(datasetAll))
    test_y = list(itemgetter(*test_idx)(labelsAll))
    test_numFrames = list(itemgetter(*test_idx)(numFramesAll))

    test_dataset = ViolenceDataset(dataset=test_x,
                                    labels=test_y,
                                    numFrames=test_numFrames,
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
    # test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=True, num_workers=4)
    indices =  [0, 1, 2, 9, 11, 3, 5]  # select your indices here as a list  
    subset = torch.utils.data.Subset(test_dataset, indices)
    dataloader = torch.utils.data.DataLoader(subset, batch_size = 1, shuffle =False)


    for data in dataloader:
        
        inputs, labels, v_names, _, paths = data

        frames_rgb = []
        for pt in paths[0]:
            image_path = pt[0]
            im=cv2.imread(image_path)
            frames_rgb.append(im)
        dyn_img, _ = getDynamicImage(frames_rgb)
        dyn_img = dyn_img.convert("RGB")
        dyn_img = np.array(dyn_img)

        print('\ninput size=', inputs.size())
        print('v_names=',v_names)
        print('image_path=',image_path)

        # image = cv2.imread(image_path)
        compute_CAM(model_, inputs, dyn_img)


def __my_main__():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelType", type=str, default="alexnet", help="model")
    parser.add_argument("--dataset", nargs='+', type=str)
    parser.add_argument("--numEpochs",type=int,default=30)
    parser.add_argument("--batchSize",type=int,default=64)
    parser.add_argument("--freezeConvLayers",type=lambda x: (str(x).lower() == 'true'), default=False, help="to fine tunning")
    parser.add_argument("--numDynamicImagesPerVideo", type=int)
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
    # parser.add_argument("--segmentPreprocessing", type=lambda x: (str(x).lower() == 'true'), default=False)

    parser.add_argument("--modelPath", type=str, default=None)
    parser.add_argument("--testDataset",type=str, default=None)
    parser.add_argument("--transferModel", type=str, default=None)

    args = parser.parse_args()

    if args.splitType == 'cam':
         __weakly_localization__()
         return 0

    if args.splitType == 'openSet-train' or args.splitType == 'openSet-test':

        openSet_experiments(mode=args.splitType, args=args)
        return 0

    input_size = 224
    shuffle = True
    if args.dataset[0] == 'rwf-2000':
        datasetAll = []
    else:
        datasetAll, labelsAll, numFramesAll, transforms = base_dataset(args.dataset[0])
    cv_test_accs = []
    cv_test_losses = []
    cv_final_epochs = []
    # patience = 5
    folds_number = 5
    fold = 0
    checkpoint_path = None
    config = None
    print(args.dataset)
       
    for train_idx, test_idx in customize_kfold(n_splits=folds_number, dataset=args.dataset[0], X_len=len(datasetAll), shuffle=shuffle):
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

        if args.transferModel is not None:
            model = load_model(args.transferModel)
        else:
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
            checkpoint_path = os.path.join(constants.PATH_RESULTS, args.dataset[0].upper(), 'checkpoints', 'DYN_Stream-{}-fold={}'.format(ss,fold))
        
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


if __name__ == "__main__":
    __my_main__()
    

