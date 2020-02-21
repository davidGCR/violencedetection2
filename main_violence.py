
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



# from .AlexNet import *
from violenceDataset import *
# from trainer import *
import trainer
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
# import LOCALIZATION.localization_utils as localization_utils
# import SALIENCY.saliencyTester as saliencyTester
            

def trainFinal(dataset, hockey_path_violence, hockey_path_noviolence, path_learning_curves, path_checkpoints, modelType, numDiPerVideos, num_workers, data_transforms,
    batch_size, num_epochs, feature_extract, joinType, scheduler_type, device, criterion, folds_number, videoSegmentLength, positionSegment):
    train_lost = []
    train_acc = []
    test_lost = []
    test_acc = []
    datasetAll, labelsAll, numFramesAll = initializeDataset.createDataset(hockey_path_violence, hockey_path_noviolence, True)  #shuffle
    train_idx = np.arange(len(datasetAll))
    train_x = list(itemgetter(*train_idx)(datasetAll))
    train_y = list(itemgetter(*train_idx)(labelsAll))
    train_numFrames = list(itemgetter(*train_idx)(numFramesAll))
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
        exp_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)
        
    dataloader = initializeDataset.getTrainDataLoader(train_x, train_y, train_numFrames, data_transforms,
                                                 numDiPerVideos, batch_size, num_workers, videoSegmentLength, positionSegment)
    dataloaders_dic = {"train": dataloader}
    tr = trainer.Trainer(model, dataloaders_dic, criterion, optimizer, exp_lr_scheduler, device, num_epochs, 'modelo_violencia_final',
                            numDiPerVideos, False, 'trainfinal')
    # for inputs, labels, video_names, bbox_segments in dataloaders_dic["train"]:
    #     print('okkkkkkkkkkkkkkkkkk')

    for epoch in range(1, num_epochs + 1):
        print("----- Epoch {}/{}".format(epoch, num_epochs))
        # Train and evaluate
        epoch_loss_train, epoch_acc_train = tr.train_epoch(epoch)
        # epoch_loss_test, epoch_acc_test = tr.val_epoch(epoch)
        # exp_lr_scheduler.step(epoch_loss_test)
        train_lost.append(epoch_loss_train)
        train_acc.append(epoch_acc_train)
        # test_lost.append(epoch_loss_test)
        # test_acc.append(epoch_acc_test)


def train(datasetAll, labelsAll, numFramesAll, path_learning_curves, path_checkpoints, modelType, numDiPerVideos, num_workers, data_transforms,
    batch_size, num_epochs, feature_extract, joinType, scheduler_type, device, criterion, folds_number, videoSegmentLength, positionSegment, dataAumentation):
    
    # for numDiPerVideos in ndis: #for experiments
    train_lost = []
    train_acc = []
    test_lost = []
    test_acc = []
    
        
    # df = pd.DataFrame(list(zip(datasetAll, labelsAll, numFramesAll)), columns=['video', 'label', 'numFrames'])
    # export_csv = df.to_csv ('hockeyFigths.csv')
        # combined = list(zip(datasetAll, labelsAll, numFramesAll))
        # random.shuffle(combined)
        # datasetAll[:], labelsAll[:], numFramesAll[:] = zip(*combined) 
        # train_idx, test_idx = None, None
    # elif dataset == 'violentflows':

    print("CONFIGURATION: ", "modelType:", modelType, ", numDiPerVideos:", numDiPerVideos, ", batch_size:", batch_size, ", num_epochs:",
            num_epochs, ", feature_extract:", feature_extract, ", joinType:", joinType, ", scheduler_type: ", scheduler_type, ', dataAumentation:',
            str(dataAumentation))

    
    fold = 0
    for train_idx, test_idx in k_folds(n_splits=folds_number, subjects=1000):
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
                                                data_transforms, numDiPerVideos, batch_size, num_workers, videoSegmentLength,
                                                positionSegment)
        else:
            train_x = []
            train_y = []
            test_x = []
            test_y = []
            # # datasetAll, labelsAll = initializeDataset.createAumentedDataset(constants.PATH_HOCKEY_AUMENTED_VIOLENCE, constants.PATH_HOCKEY_AUMENTED_NON_VIOLENCE, shuffle)  #shuffle
            for idx in test_idx:
                dyImgs = os.listdir(datasetAll[idx])
                dyImgs.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
                test_example = datasetAll.pop(idx)
                test_example = os.path.join(test_example,dyImgs[0])
                test_x.append(test_example)

                test_example_label = labelsAll.pop(idx)
                # test_example_label = os.path.join(test_example_label,dyImgs[0])
                test_y.append(test_example_label)

            for i in range(len(datasetAll)):
                dyImgs = os.listdir(datasetAll[i])
                dyImgs.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
                for j,img in enumerate(dyImgs):
                    dyImgs[j]=os.path.join(datasetAll[i],img)
                    train_y.append(labelsAll[i])
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
        # data_rows = []
        # for inputs, labels, video_names, bbox_segments in dataloaders_dict["train"]:
        #     print('videos names: ', video_names, labels)

            # row = [video_name[0]+'---segment No: '+str(num_segment), iou]
            # data_rows.append(row)
        # MODEL_NAME = modelType+'-'+str(numDiPerVideos)+'-'+joinType+'-segmentLength:'+str(videoSegmentLength)+'-positionSegment:'+positionSegment+'-numEpochs:'+str(num_epochs)+'-dataAumentation:'+str(dataAumentation)
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
        # MODEL_NAME = 'probando'
        # MODEL_NAME = str(modelType) + '_Finetuned-' + str(not feature_extract) + '-' +'_di-'+str(numDiPerVideos) + '_fusionType-'+str(joinType) +'_num_epochs-' +str(num_epochs)
        MODEL_NAME = modelType+'-'+str(numDiPerVideos)+'-'+joinType+'-segmentLength:'+str(videoSegmentLength)+'-positionSegment:'+positionSegment+'-numEpochs:'+str(num_epochs)+'-dataAumentation:'+str(dataAumentation)
        # if folds_number == 1:
        #     MODEL_NAME = MODEL_NAME+constants.LABEL_PRODUCTION_MODEL
        # print('model_name: ', MODEL_NAME)

        tr = trainer.Trainer( model, dataloaders_dict, criterion, optimizer, exp_lr_scheduler, device, num_epochs, None, numDiPerVideos,False, 'train')

        for epoch in range(1, num_epochs + 1):
            print("----- Epoch {}/{}".format(epoch, num_epochs))
            # Train and evaluate
            epoch_loss_train, epoch_acc_train = tr.train_epoch(epoch)
            epoch_loss_test, epoch_acc_test = tr.val_epoch(epoch)
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
    saveLearningCurve(os.path.join(path_learning_curves,MODEL_NAME+"-val_lost.txt"), test_lost)
    saveLearningCurve(os.path.join(path_learning_curves,MODEL_NAME+"-val_acc.txt"),test_acc)
    # saveList(path_learning_curves, modelType, scheduler_type, "train_lost", numDiPerVideos, dataset_source, feature_extract, joinType, train_lost,)
    # saveList(path_learning_curves, modelType, scheduler_type,"train_acc",numDiPerVideos, dataset_source, feature_extract, joinType, train_acc, )
    # saveList(path_learning_curves, modelType, scheduler_type, "test_lost", numDiPerVideos, dataset_source, feature_extract, joinType, test_lost, )
    # saveList(path_learning_curves, modelType, scheduler_type, "test_acc", numDiPerVideos, dataset_source, feature_extract, joinType, test_acc, )

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
    parser.add_argument("--typeTrain", type=str)
    parser.add_argument("--dataAumentation",type=lambda x: (str(x).lower() == 'true'), default=False)

    args = parser.parse_args()
    typeTrain = args.typeTrain
    
    path_learning_curves = constants.PATH_VIOLENCE_LEARNING_CURVES
    path_checkpoints = constants.PATH_VIOLENCE_CHECKPOINTS

    modelType = args.modelType
    batch_size = args.batchSize
    num_epochs = args.numEpochs
    feature_extract = args.featureExtract
    joinType = args.joinType
    scheduler_type = args.schedulerType
    numDynamicImagesPerVideo = args.numDynamicImagesPerVideo
    
    folds_number = args.foldsNumber
    num_workers = args.numWorkers
    input_size = 224
    videoSegmentLength = args.videoSegmentLength
    positionSegment = args.positionSegment
    dataAumentation = args.dataAumentation

    transforms = createTransforms(input_size)
    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()
    # path_models = '/media/david/datos/Violence DATA/violentflows/Models/'
    # path_results = '/media/david/datos/Violence DATA/violentflows/Results/'+dataset_source
    # gpath = '/media/david/datos/Violence DATA/violentflows/movies Frames'
    shuffle = True
    if typeTrain == 'final':
        trainFinal(dataset, constants.PATH_HOCKEY_FRAMES_VIOLENCE, constants.PATH_HOCKEY_FRAMES_NON_VIOLENCE, path_learning_curves, path_checkpoints, modelType, numDynamicImagesPerVideo, num_workers, transforms,
            batch_size, num_epochs, feature_extract, joinType, scheduler_type, device, criterion, folds_number,videoSegmentLength, positionSegment)
    elif typeTrain == 'train':
        if not dataAumentation:
            datasetAll, labelsAll, numFramesAll = initializeDataset.createDataset(constants.PATH_HOCKEY_FRAMES_VIOLENCE, constants.PATH_HOCKEY_FRAMES_NON_VIOLENCE, shuffle)  #shuffle
        else:
            shuffle = True
            datasetAll, labelsAll, _ = initializeDataset.createDataset(constants.PATH_HOCKEY_AUMENTED_VIOLENCE, constants.PATH_HOCKEY_AUMENTED_NON_VIOLENCE, shuffle)  #shuffle
            # print(datasetAll[:5])
            # print(labelsAll[:5])
        train(datasetAll, labelsAll, None, path_learning_curves, path_checkpoints, modelType, numDynamicImagesPerVideo, num_workers, transforms,
            batch_size, num_epochs, feature_extract, joinType, scheduler_type, device, criterion, folds_number,videoSegmentLength, positionSegment, dataAumentation)
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

