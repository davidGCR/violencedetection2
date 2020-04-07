
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



def train(trainMode, datasetAll, labelsAll, numFramesAll, path_learning_curves, path_checkpoints, modelType, numDiPerVideos, num_workers, data_transforms,
    batch_size, num_epochs, feature_extract, joinType, scheduler_type, device, criterion, folds_number, videoSegmentLength, positionSegment,
    dataAumentation, overlaping, frame_skip):
    
    # for numDiPerVideos in ndis: #for experiments
    train_errors = []
    train_acc = []
    
        
    # df = pd.DataFrame(list(zip(datasetAll, labelsAll, numFramesAll)), columns=['video', 'label', 'numFrames'])
    # export_csv = df.to_csv ('hockeyFigths.csv')
        # combined = list(zip(datasetAll, labelsAll, numFramesAll))
        # random.shuffle(combined)
        # datasetAll[:], labelsAll[:], numFramesAll[:] = zip(*combined) 
        # train_idx, test_idx = None, None
    # elif dataset == 'violentflows':

    print(trainMode, "--- CONFIGURATION: ", "modelType:", modelType, ", numDiPerVideos:", numDiPerVideos, ", batch_size:", batch_size, ", num_epochs:",
            num_epochs, ", feature_extract:", feature_extract, ", joinType:", joinType, ", scheduler_type: ", scheduler_type, ', dataAumentation:',
            str(dataAumentation), ', overlapping:', str(overlaping), 'frame_skip: ',str(frame_skip))

    if trainMode == 'validationMode':
        fold = 0
        test_cv_acc = []
        
        for train_idx, test_idx in k_folds(n_splits=folds_number, subjects=len(datasetAll)):
        # for dataset_train, dataset_train_labels,dataset_test,dataset_test_labels   in k_folds_from_folders(vif_path, 5):
            fold = fold + 1
            print("**************** Fold:{}/{} ".format(fold, folds_number))
            train_x, train_y, test_x, test_y = None, None, None, None
            # print('fold: ',len(train_idx),len(test_idx))
            
                
            # train_x, train_y, train_numFrames, test_x, test_y, test_numFrames, data_transforms, numDiPerVideos, batch_size, num_workers, videoSegmentLength, positionSegmen
            if not dataAumentation:
                train_x = list(itemgetter(*train_idx)(datasetAll))
                train_y = list(itemgetter(*train_idx)(labelsAll))
                train_numFrames = list(itemgetter(*train_idx)(numFramesAll))
                test_x = list(itemgetter(*test_idx)(datasetAll))
                test_y = list(itemgetter(*test_idx)(labelsAll))
                test_numFrames = list(itemgetter(*test_idx)(numFramesAll))
                initializeDataset.print_balance(train_y, test_y)
                dataloaders_dict = initializeDataset.getDataLoaders(train_x, train_y, train_numFrames, test_x, test_y, test_numFrames,
                                                    data_transforms, numDiPerVideos, train_batch_size=batch_size, test_batch_size=1,
                                                    train_num_workers=num_workers, test_num_workers=1, videoSegmentLength=videoSegmentLength,
                                                    positionSegment=positionSegment, overlaping=overlaping, frame_skip=frame_skip)
            else:
                train_x = []
                train_y = []
                test_x = []
                test_y = []
                # # datasetAll, labelsAll = initializeDataset.createAumentedDataset(constants.PATH_HOCKEY_AUMENTED_VIOLENCE, constants.PATH_HOCKEY_AUMENTED_NON_VIOLENCE, shuffle)  #shuffle
                # datasetAll_cpy = copy.copy(datasetAll)
                # labelsAll_cpy = copy.copy(labelsAll)
                # print(test_idx)
                # MODEL_NAME = 'Using-'
                for idx in test_idx:
                    # print(len(datasetAll), idx)
                    dyImgs = os.listdir(datasetAll[idx])
                    dyImgs.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
                    test_example = datasetAll[idx]
                    test_example = os.path.join(test_example,dyImgs[1])
                    test_x.append(test_example)

                    test_example_label = labelsAll[idx]
                    # test_example_label = os.path.join(test_example_label,dyImgs[0])
                    test_y.append(test_example_label)

                for idx in train_idx:
                    dyImgs = os.listdir(datasetAll[idx])
                    dyImgs.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
                    for j,img in enumerate(dyImgs):
                        dyImgs[j]=os.path.join(datasetAll[idx],img)
                        train_y.append(labelsAll[idx])
                    train_x.extend(dyImgs)
                
                combined = list(zip(train_x, train_y))
                random.shuffle(combined)
                train_x[:], train_y[:] = zip(*combined)

                combined = list(zip(test_x, test_y))
                random.shuffle(combined)
                test_x[:], test_y[:] = zip(*combined)
                # df = pd.DataFrame(list(zip(*[test_x, test_y]))).add_prefix('Col')
                # df.to_csv('testAumented.csv', index=False)
                # df = pd.DataFrame(list(zip(*[train_x, train_y]))).add_prefix('Col')
                # df.to_csv('trainAumented.csv', index=False)
                
                dataloaders_dict = initializeDataset.getDataLoadersAumented(train_x, train_y, test_x, test_y, data_transforms, batch_size, num_workers)
            model, input_size = initialize_model( model_name=modelType, num_classes=2, feature_extract=feature_extract, numDiPerVideos=numDiPerVideos, joinType=joinType, use_pretrained=True)
            model.to(device)
            params_to_update = verifiParametersToTrain(model, feature_extract)
            # Observe that all parameters are being optimized
            optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
            # Decay LR by a factor of 0.1 every 7 epochs
            if scheduler_type == "StepLR":
                exp_lr_scheduler = lr_scheduler.StepLR( optimizer, step_size=7, gamma=0.1 )
            elif scheduler_type == "OnPlateau":
                exp_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau( optimizer, patience=5, verbose=True )
            MODEL_NAME = modelType+'-'+str(numDiPerVideos)+'-Finetuned:'+str(not feature_extract)+'-'+joinType+'-segmentLength:'+str(videoSegmentLength)+'-positionSegment:'+positionSegment+'-numEpochs:'+str(num_epochs)+'-dataAumentation:'+str(dataAumentation)+'-overlaping:'+str(overlaping)+'-skipFrame:'+str(frame_skip)
            checkpointPath = os.path.join(constants.PATH_VIOLENCE_CHECKPOINTS, MODEL_NAME)
            tr = trainer.Trainer(model, dataloaders_dict, criterion, optimizer, exp_lr_scheduler, device,
                                num_epochs, checkpointPath, numDiPerVideos,False, 'train', save_model=False)

            for epoch in range(1, num_epochs + 1):
                print("----- Epoch {}/{}".format(epoch, num_epochs))
                # Train and evaluate
                epoch_loss_train, epoch_acc_train = tr.train_epoch(epoch)
                # epoch_loss_test, epoch_acc_test = tr.val_epoch(epoch)
                exp_lr_scheduler.step(epoch_loss_train)
                train_errors.append(epoch_loss_train)
                train_acc.append(epoch_acc_train)
                # test_lost.append(epoch_loss_test)
                # test_acc.append(epoch_acc_test)
            ########## TESTING ##########
            tester = Tester(model=tr.getModel(), dataloader=dataloaders_dict['val'], loss=criterion, numDiPerVideos=numDiPerVideos, device=device)
            predictions, scores, gt_labels, test_error, preprocess_times, inference_times = tester.test_model()
            pairs = {'pp_time': preprocess_times, 'inf_time': inference_times}
            df = pd.DataFrame.from_dict(pairs)
            if not os.path.exists(constants.PATH_TIME_RESULTS):
                os.makedirs(constants.PATH_TIME_RESULTS)
            if not os.path.exists(os.path.join(constants.PATH_TIME_RESULTS, MODEL_NAME)):
                os.makedirs(os.path.join(constants.PATH_TIME_RESULTS, MODEL_NAME))
            df.to_csv(os.path.join(constants.PATH_TIME_RESULTS, MODEL_NAME, 'fold-'+str(fold)+'.csv'))
            

            acc = cv_it_accuracy(predictions,gt_labels)
            # print('accuracy tests: ', acc)
            test_cv_acc.append(acc)
            timeCost(os.path.join(constants.PATH_TIME_RESULTS, MODEL_NAME))
            
        
        avg_acc = np.average(test_cv_acc)
        print('K-Folds accuracies: ', test_cv_acc)
        print('K-folds Avg accuracy: ', avg_acc)
        # print('Test error: ', test_error)
        # print('Folds test FPS: ', test_fps, np.average(test_fps))
        # print('Test error: ', test_error)

        # saveLearningCurve(os.path.join(path_learning_curves, MODEL_NAME + "-gtruth.txt"), gt_labels)
        # saveLearningCurve(os.path.join(path_learning_curves, MODEL_NAME + "-predictions.txt"), predictions)
        # saveLearningCurve(os.path.join(path_learning_curves, MODEL_NAME + "-train_error.txt"), train_errors)
        # saveLearningCurve(os.path.join(path_learning_curves, MODEL_NAME + "-train_acc.txt"), train_acc)
        
    elif trainMode == 'finalMode':
        model, input_size = initialize_model( model_name=modelType, num_classes=2, feature_extract=feature_extract, numDiPerVideos=numDiPerVideos, joinType=joinType, use_pretrained=True)
        model.to(device)
        params_to_update = verifiParametersToTrain(model, feature_extract)
        # Observe that all parameters are being optimized
        optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
        # Decay LR by a factor of 0.1 every 7 epochs
        if scheduler_type == "StepLR":
            exp_lr_scheduler = lr_scheduler.StepLR( optimizer, step_size=7, gamma=0.1 )
        elif scheduler_type == "OnPlateau":
            exp_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau( optimizer, patience=5, verbose=True )
        MODEL_NAME = modelType+'-'+str(numDiPerVideos)+'-Finetuned:'+str(not feature_extract)+'-'+joinType+'-segmentLength:'+str(videoSegmentLength)+'-positionSegment:'+positionSegment+'-numEpochs:'+str(num_epochs)+'-dataAumentation:'+str(dataAumentation)+'-overlaping:'+str(overlaping)
        
        checkpointPath = os.path.join(constants.PATH_VIOLENCE_CHECKPOINTS, MODEL_NAME)
        tr = trainer.Trainer(model, None, criterion, optimizer, exp_lr_scheduler, device,
                            num_epochs, checkpointPath, numDiPerVideos, False, 'train', save_model=False)
        
        trainFinalModel(model_name=MODEL_NAME, my_trainer=tr, datasetAll=datasetAll, labelsAll=labelsAll, numFramesAll=numFramesAll,
                        numDiPerVideos=numDiPerVideos, positionSegment=positionSegment, overlaping=overlaping,
                        videoSegmentLength=videoSegmentLength, ttransform=data_transforms["train"], batch_size=batch_size,
                        num_epochs=num_epochs, num_workers=num_workers)

def timeCost(path):
    folds_times = os.listdir(path)
    folds_times.sort()
    ppTimer = FPSMeter()
    infTimer = FPSMeter()
    for i, fold in enumerate(folds_times):
        data = pd.read_csv(os.path.join(path, fold))
        for index, row in data.iterrows():
            ppTimer.update(row['pp_time'])
            infTimer.update(row['inf_time'])
        shape = data.shape
        print('Fold: %s, Number examples: %s'%(str(i+1), str(shape[0])))
        ppTimer.print_statistics()
        infTimer.print_statistics()

   
def __main__():

    # python3 main.py --dataset hockey --numEpochs 12 --ndis 1 --foldsNumber 1 --featureExtract true --checkpointPath BlackBoxModels
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelType",type=str,default="alexnet",help="model")
    parser.add_argument("--numEpochs",type=int,default=30)
    parser.add_argument("--batchSize",type=int,default=64)
    parser.add_argument("--featureExtract",type=lambda x: (str(x).lower() == 'true'), default=False, help="to fine tunning")
    parser.add_argument("--schedulerType",type=str,default="OnPlateau",help="learning rate scheduler")
    parser.add_argument("--numDynamicImagesPerVideo", type=int)
    parser.add_argument("--joinType", type=str)
    parser.add_argument("--foldsNumber", type=int, default=5)
    parser.add_argument("--numWorkers", type=int, default=4)
    parser.add_argument("--videoSegmentLength", type=int)
    parser.add_argument("--positionSegment", type=str)
    parser.add_argument("--trainMode", type=str)
    parser.add_argument("--overlaping", type=float)
    parser.add_argument("--frameSkip", type=int, default=0)
    parser.add_argument("--dataAumentation",type=lambda x: (str(x).lower() == 'true'), default=False)

    args = parser.parse_args()
    trainMode = args.trainMode
    
    path_learning_curves = constants.PATH_VIOLENCE_LEARNING_CURVES
    path_checkpoints = constants.PATH_VIOLENCE_CHECKPOINTS

    modelType = args.modelType
    batch_size = args.batchSize
    num_epochs = args.numEpochs
    frameSkip = args.frameSkip

    
    feature_extract = args.featureExtract
    joinType = args.joinType
    scheduler_type = args.schedulerType
    numDynamicImagesPerVideo = args.numDynamicImagesPerVideo
    overlaping = args.overlaping
    # print('overlaping: ', overlaping, type(overlaping))
    
    folds_number = args.foldsNumber
    num_workers = args.numWorkers
    input_size = 224
    videoSegmentLength = args.videoSegmentLength
    positionSegment = args.positionSegment
    dataAumentation = args.dataAumentation

    transforms = createTransforms(input_size)
    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device: ', device)
    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()
    # path_models = '/media/david/datos/Violence DATA/violentflows/Models/'
    # path_results = '/media/david/datos/Violence DATA/violentflows/Results/'+dataset_source
    # gpath = '/media/david/datos/Violence DATA/violentflows/movies Frames'
    shuffle = True
    path_violence = constants.PATH_HOCKEY_FRAMES_VIOLENCE
    path_non_violence = constants.PATH_HOCKEY_FRAMES_NON_VIOLENCE
    
    if not dataAumentation:
        datasetAll, labelsAll, numFramesAll = initializeDataset.createDataset(path_violence, path_non_violence, shuffle)  #shuffle
        train(trainMode,datasetAll, labelsAll, numFramesAll, path_learning_curves, path_checkpoints, modelType, numDynamicImagesPerVideo, num_workers, transforms,
        batch_size, num_epochs, feature_extract, joinType, scheduler_type, device, criterion, folds_number, videoSegmentLength, positionSegment, dataAumentation,
        overlaping, frameSkip)
    else:
        shuffle = True
        datasetAll, labelsAll, _ = initializeDataset.createDataset(constants.PATH_HOCKEY_AUMENTED_VIOLENCE, constants.PATH_HOCKEY_AUMENTED_NON_VIOLENCE, shuffle)  #shuffle
        # print(datasetAll[:5])
        # print(labelsAll[:5])
        train(trainMode, datasetAll, labelsAll, None, path_learning_curves, path_checkpoints, modelType, numDynamicImagesPerVideo, num_workers, transforms,
            batch_size, num_epochs, feature_extract, joinType, scheduler_type, device, criterion, folds_number, videoSegmentLength, positionSegment,
            dataAumentation, overlaping, frameSkip)
    # num_classes = 2
    # saliency_model_file = 'd'
    # input_size = (224,224)
    # saliency_model_file = 'SALIENCY/Models/anomaly/mask_model_10_frames_di__epochs-12.tar'
    # threshold = 0.5
    # saliency_tester = saliencyTester.SaliencyTester(saliency_model_file, num_classes, None, test_names,
    #                                     input_size, saliency_model_config, ndis, None)
    # typePersonDetector = 'yolov3'
    # only_video_name = None
    # plot = True
    # h = 288
    # w = 360
    # overlapping = 0.5
    # datasetAll, labelsAll, numFramesAll = initializeDataset.createDataset(path_violence, path_noviolence, suffle=False)
    # dataloader, violenceDataset = getOnlineDataLoader(datasetAll, labelsAll, numFramesAll, transform, numDiPerVideos, batch_size, num_workers, overlapping)

    # online(violenceDataset, saliency_tester, type_person_detector, h, w, plot, only_video_name)

__main__()

