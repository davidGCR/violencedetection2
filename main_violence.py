
import sys
# sys.path.insert(1, '/media/david/datos/PAPERS-SOURCE_CODE/violencedetection')
# sys.path.insert(1,'/Users/davidchoqueluqueroman/Desktop/PAPERS-CODIGOS/violencedetection2')
import include
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
import copy



# from .AlexNet import *
from violenceDataset import *
# from trainer import *
import trainer
from tester import Tester
from kfolds import *
from operator import itemgetter
import random
from initializeModel import *
from util import *
from parameters import *
from transforms import *
# from MaskDataset import MaskDataset
# from saliency_model import *
import initializeDataset
import constants
import pandas as pd

from FPS import FPSMeter
from torch.utils.tensorboard import SummaryWriter
import util
# import LOCALIZATION.localization_utils as localization_utils
# import SALIENCY.saliencyTester as saliencyTester
            

def trainFinalModel(model_name, my_trainer, datasetAll, labelsAll, numFramesAll, numDiPerVideos, positionSegment,
                overlaping,videoSegmentLength, ttransform, batch_size, num_epochs, num_workers):
    train_lost = []
    train_acc = []

    datasett = ViolenceDataset(dataset=datasetAll, labels=labelsAll, numFrames=numFramesAll,
            spatial_transform=ttransform, numDynamicImagesPerVideo=numDiPerVideos,
         videoSegmentLength= videoSegmentLength, positionSegment = positionSegment, overlaping=overlaping )
        
    

    dataloader = torch.utils.data.DataLoader( datasett, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    dataloaders_dic = {'train': dataloader,
                        'val': None}
    my_trainer.dataloaders = dataloaders_dic
    checkpointPath = checkpointPath = os.path.join(constants.PATH_VIOLENCE_CHECKPOINTS, model_name)
    my_trainer.checkpoint_path = checkpointPath
    my_trainer.save_model = True
   

    for epoch in range(1, num_epochs + 1):
        print("----- Epoch {}/{}".format(epoch, num_epochs))
        # Train and evaluate
        epoch_loss_train, epoch_acc_train = my_trainer.train_epoch(epoch)
        # epoch_loss_test, epoch_acc_test = tr.val_epoch(epoch)
        # exp_lr_scheduler.step(epoch_loss_test)
        train_lost.append(epoch_loss_train)
        train_acc.append(epoch_acc_train)
        # test_lost.append(epoch_loss_test)
        # test_acc.append(epoch_acc_test)

def cv_it_accuracy(predictions, gt_labels):
    running_corrects = np.sum(predictions == gt_labels)
    acc = running_corrects / gt_labels.shape[0]
    return acc

def modelSelection(datasetAll, labelsAll, numFramesAll, path_learning_curves, path_checkpoints, modelType, numDiPerVideos, num_workers, data_transforms,
    batch_size, num_epochs, feature_extract, joinType, scheduler_type, device, criterion, folds_number, videoSegmentLength, positionSegment,
    dataAumentation, overlaping, frame_skip):
    dataloaders_dict = initializeDataset.getDataLoaders(train_x, train_y, train_numFrames, test_x, test_y, test_numFrames,
                                                    data_transforms, numDiPerVideos, train_batch_size=batch_size, test_batch_size=1,
                                                    train_num_workers=num_workers, test_num_workers=1, videoSegmentLength=videoSegmentLength,
                                                    positionSegment=positionSegment, overlaping=overlaping, frame_skip=frame_skip)


        
def timeCostKfolds(folds_df):
    # folds_times = os.listdir(path)
    # folds_times.sort()
    folds_pp_times = []
    folds_inf_times = []
    
    for i, fold_df in enumerate(folds_df):
        ppTimer = FPSMeter()
        infTimer = FPSMeter()
        # data = pd.read_csv(os.path.join(path, fold))
        for index, row in fold_df.iterrows():
            ppTimer.update(row['pp_time'])
            infTimer.update(row['inf_time'])
        folds_pp_times.append(ppTimer.mspf())
        folds_inf_times.append(infTimer.mspf())
        
        # shape = fold_df.shape
        # print('Fold: %s, Number examples: %s'%(str(i+1), str(shape[0])))
        # ppTimer.print_statistics()
        # infTimer.print_statistics()
    avg_mspf_pp = np.average(folds_pp_times)
    avg_mspf_inf = np.average(folds_inf_times)
    
    return avg_mspf_pp, avg_mspf_inf
    

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
    parser.add_argument("--overlaping", type=float)
    parser.add_argument("--frameSkip", type=int, default=0)

    args = parser.parse_args()
    split_type = args.split_type
    
    path_learning_curves = constants.PATH_VIOLENCE_LEARNING_CURVES
    path_checkpoints = constants.PATH_VIOLENCE_CHECKPOINTS

    modelType = args.modelType
    batch_size = args.batchSize
    num_epochs = args.numEpochs
    frame_skip = args.frameSkip
    feature_extract = args.featureExtract
    joinType = args.joinType
    numDynamicImagesPerVideo = args.numDynamicImagesPerVideo
    overlaping = args.overlaping
    num_workers = args.numWorkers
    input_size = 224
    videoSegmentLength = args.videoSegmentLength
    positionSegment = args.positionSegment
    shuffle = True
    path_violence = constants.PATH_HOCKEY_FRAMES_VIOLENCE
    path_non_violence = constants.PATH_HOCKEY_FRAMES_NON_VIOLENCE
            
    datasetAll, labelsAll, numFramesAll = initializeDataset.createDataset(path_violence, path_non_violence, shuffle)  #shuffle
    transforms = createTransforms(input_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if split_type == 'train-test':
        folds_number = 1
        if not os.path.exists(os.path.join(constants.PATH_HOCKEY_README, 'train_split.csv')):
            for train_idx, test_idx in k_folds(n_splits=folds_number, subjects=len(datasetAll)):
                # print("**************** Fold:{}/{} ".format(fold, folds_number))
                train_x, train_y, test_x, test_y = None, None, None, None
            
                train_x = list(itemgetter(*train_idx)(datasetAll))
                train_y = list(itemgetter(*train_idx)(labelsAll))
                train_numFrames = list(itemgetter(*train_idx)(numFramesAll))
                test_x = list(itemgetter(*test_idx)(datasetAll))
                test_y = list(itemgetter(*test_idx)(labelsAll))
                test_numFrames = list(itemgetter(*test_idx)(numFramesAll))
                # initializeDataset.print_balance(train_y, test_y)

                train_split = zip(train_x, train_y, train_numFrames)
                test_split = zip(test_x, test_y, test_numFrames)
                out_file = os.path.join(constants.PATH_HOCKEY_README,'train_split.csv')
                with open(out_file, 'w') as f:
                    writer = csv.writer(f, delimiter='\t')
                    writer.writerows(train_split)
        
                out_file = os.path.join(constants.PATH_HOCKEY_README, 'test_split.csv')
                with open(out_file, 'w') as f:
                    writer = csv.writer(f, delimiter='\t')
                    writer.writerows(test_split)
        else:
            train_x = []
            train_y = []
            test_x = []
            test_y = []
            train_numFrames = []
            test_numFrames = []
            
            with open(os.path.join(constants.PATH_HOCKEY_README,'train_split.csv')) as csvfile:
                readCSV = csv.reader(csvfile, delimiter='\t')
                for row in readCSV:
                    train_x.append(row[0])
                    train_y.append(int(row[1]))
                    train_numFrames.append(int(row[2]))
            
            with open(os.path.join(constants.PATH_HOCKEY_README,'test_split.csv')) as csvfile:
                readCSV = csv.reader(csvfile, delimiter='\t')
                for row in readCSV:
                    test_x.append(row[0])
                    test_y.append(int(row[1]))
                    test_numFrames.append(int(row[2]))

        initializeDataset.print_balance(train_y, test_y)

        dataloaders_dict = initializeDataset.getDataLoaders(train_x, train_y, train_numFrames, test_x, test_y, test_numFrames,
                                                transforms, numDynamicImagesPerVideo, train_batch_size=batch_size, test_batch_size=1,
                                                train_num_workers=num_workers, test_num_workers=1, videoSegmentLength=videoSegmentLength,
                                                positionSegment=positionSegment, overlaping=overlaping, frame_skip=frame_skip)
            
        model, _ = initialize_model(model_name=modelType,
                                num_classes=2,
                                feature_extract=feature_extract,
                                numDiPerVideos=numDynamicImagesPerVideo,
                                joinType=joinType,
                                use_pretrained=True)
        model.to(device)
        params_to_update = verifiParametersToTrain(model, feature_extract)
        optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        criterion = nn.CrossEntropyLoss()
        # split_type = split_type+str(fold)
        experimentConfig = 'HOCKEY-Model-{}, segmentLen-{}, numDynIms-{}, frameSkip-{}, epochs-{}, split_type-{}'.format(modelType,
                                                                                                videoSegmentLength,
                                                                                                numDynamicImagesPerVideo,
                                                                                                frame_skip,
                                                                                                num_epochs,
                                                                                                split_type)
        checkpoint_path = os.path.join(constants.HOCKEY_PATH_CHECKPOINTS, experimentConfig + '.tar')
        
        writer = SummaryWriter('runs/'+experimentConfig)
        tr = trainer.Trainer(model=model,
                            train_dataloader=dataloaders_dict['train'],
                            val_dataloader=dataloaders_dict['val'],
                            criterion=criterion,
                            optimizer=optimizer,
                            scheduler=exp_lr_scheduler,
                            device=device,
                            num_epochs=num_epochs,
                            checkpoint_path=checkpoint_path,
                            numDynamicImage=numDynamicImagesPerVideo,
                            plot_samples=False,
                            train_type='train',
                            save_model=False)

        for epoch in range(1, num_epochs + 1):
            print("----- Epoch {}/{}".format(epoch, num_epochs))
            # Train and evaluate
            epoch_loss_train, epoch_acc_train = tr.train_epoch(epoch)
            epoch_loss_val, epoch_acc_val = tr.val_epoch(epoch)
            exp_lr_scheduler.step()

            writer.add_scalar('training loss', epoch_loss_train, epoch)
            writer.add_scalar('validation loss', epoch_loss_val, epoch)
            writer.add_scalar('training Acc', epoch_acc_train, epoch)
            writer.add_scalar('validation Acc', epoch_acc_val, epoch)

            
    elif split_type == 'cross_val':
        folds_number = 5
        fold = 0
        for train_idx, test_idx in k_folds(n_splits=folds_number, subjects=len(datasetAll)):
            fold = fold + 1
            print("**************** Fold:{}/{} ".format(fold, folds_number))
            train_x, train_y, test_x, test_y = None, None, None, None
        
            train_x = list(itemgetter(*train_idx)(datasetAll))
            train_y = list(itemgetter(*train_idx)(labelsAll))
            train_numFrames = list(itemgetter(*train_idx)(numFramesAll))
            test_x = list(itemgetter(*test_idx)(datasetAll))
            test_y = list(itemgetter(*test_idx)(labelsAll))
            test_numFrames = list(itemgetter(*test_idx)(numFramesAll))
            initializeDataset.print_balance(train_y, test_y)

            dataloaders_dict = initializeDataset.getDataLoaders(train_x, train_y, train_numFrames, test_x, test_y, test_numFrames,
                                                transforms, numDynamicImagesPerVideo, train_batch_size=batch_size, test_batch_size=1,
                                                train_num_workers=num_workers, test_num_workers=1, videoSegmentLength=videoSegmentLength,
                                                positionSegment=positionSegment, overlaping=overlaping, frame_skip=frame_skip)
            
            model, _ = initialize_model(model_name=modelType,
                                    num_classes=2,
                                    feature_extract=feature_extract,
                                    numDiPerVideos=numDynamicImagesPerVideo,
                                    joinType=joinType,
                                    use_pretrained=True)
            model.to(device)
            params_to_update = verifiParametersToTrain(model, feature_extract)
            optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
            
            criterion = nn.CrossEntropyLoss()
            split_type = split_type+str(fold)
            experimentConfig = 'HOCKEY-Model-{}, segmentLen-{}, numDynIms-{}, frameSkip-{}, epochs-{}, split_type-{}'.format(modelType,
                                                                                                    videoSegmentLength,
                                                                                                    numDynamicImagesPerVideo,
                                                                                                    frame_skip,
                                                                                                    num_epochs,
                                                                                                    split_type)
            writer = SummaryWriter('runs/'+experimentConfig)
            tr = trainer.Trainer(model=model,
                                train_dataloader=dataloaders_dict['train'],
                                val_dataloader=dataloaders_dict['val'],
                                criterion=criterion,
                                optimizer=optimizer,
                                scheduler=exp_lr_scheduler,
                                device=device,
                                num_epochs=num_epochs,
                                checkpoint_path=experimentConfig,
                                numDynamicImage=numDynamicImagesPerVideo,
                                plot_samples=False,
                                train_type='train',
                                save_model=False)

            for epoch in range(1, num_epochs + 1):
                print("Fold {} ----- Epoch {}/{}".format(fold,epoch, num_epochs))
                # Train and evaluate
                epoch_loss_train, epoch_acc_train = tr.train_epoch(epoch)
                epoch_loss_val, epoch_acc_val = tr.val_epoch(epoch)
                exp_lr_scheduler.step()

                writer.add_scalar('training loss', epoch_loss_train, epoch)
                writer.add_scalar('validation loss', epoch_loss_val, epoch)
                writer.add_scalar('training Acc', epoch_acc_train, epoch)
                writer.add_scalar('validation Acc', epoch_acc_val, epoch)
    

__main__()

