import sys
# sys.path.insert(1, '/media/david/datos/PAPERS-SOURCE_CODE/violencedetection')
sys.path.insert(1,'/Users/davidchoqueluqueroman/Desktop/PAPERS-CODIGOS/violencedetection2')
import anomalyDataset
import os
import re
from util import video2Images2, saveList, get_model_name
import csv
import pandas as pd
import numpy as np
import torch.nn as nn
import torch
from initializeModel import initialize_model
from parameters import verifiParametersToTrain
import transforms_anomaly
import torch.optim as optim
from torch.optim import lr_scheduler
from trainer import Trainer
import random
import util
# import initializeDataset
import constants
import glob
import argparse
import anomalyInitializeDataset
from tester import Tester
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt

def testing(model, dataloaders, device, numDiPerVideos, plot_samples):
    tester = Tester(model, dataloaders, device, numDiPerVideos, plot_samples)
    gt_labels, predictions, scores = tester.test_model()
    gt_labels = np.array(gt_labels)
    predictions = np.array(predictions)
    scores = np.array(scores)
    # print(predictions, predictions.shape)
    # print(type(scores), scores.shape)
    scores = scores[:,1]
    # print(scores)
    
    auc_0 = roc_auc_score(gt_labels, scores)
    fpr, tpr, thresholds = roc_curve(gt_labels,scores)
    roc_auc = auc(fpr, tpr)
    print('fpr, tpr:', len(fpr), len(tpr))
    print('thresholds: ', thresholds)
    print('auc_0, roc_auc: ',auc_0, roc_auc)
    # print('fpr: ', fpr)
    # print('tpr: ', tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

    return auc

def training(modelType, num_classes, feature_extract, numDiPerVideos, joinType, device, additional_info, path_learning_curves, 
                scheduler_type, num_epochs, dataloaders_dict, path_checkpoints, plot_samples, operation):
    model, input_size = initialize_model( model_name=modelType, num_classes=num_classes, feature_extract=feature_extract, numDiPerVideos=numDiPerVideos, 
                                            joinType=joinType, use_pretrained=True)
    print(model)
    model.to(device)

    MODEL_NAME = util.get_model_name(modelType, scheduler_type, numDiPerVideos, feature_extract, joinType, num_epochs)
    MODEL_NAME += additional_info
    MODEL_NAME = MODEL_NAME+'-FINAL' if operation == constants.OPERATION_TRAINING_FINAL else MODEL_NAME
    print(MODEL_NAME)

    params_to_update = verifiParametersToTrain(model, feature_extract)
    optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    # Decay LR by a factor of 0.1 every 7 epochs
    if scheduler_type == "StepLR":
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    elif scheduler_type == "OnPlateau":
        exp_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau( optimizer, patience=5, verbose=True )
    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(model, dataloaders_dict, criterion, optimizer, exp_lr_scheduler, device, num_epochs,
                       os.path.join(path_checkpoints,MODEL_NAME), numDiPerVideos, plot_samples, operation)
    train_lost = []
    train_acc = []
    val_lost = []
    val_acc = []
    for epoch in range(1, num_epochs + 1):
        print("----- Epoch {}/{}".format(epoch, num_epochs))
        # Train and evaluate
        if operation == constants.OPERATION_TRAINING_FINAL:
            epoch_loss_train, epoch_acc_train = trainer.train_epoch(epoch)
            train_lost.append(epoch_loss_train)
            train_acc.append(epoch_acc_train)
            exp_lr_scheduler.step(epoch_loss_train)
            

        elif operation == constants.OPERATION_TRAINING or operation == constants.OPERATION_TRAINING_AUMENTED:
            epoch_loss_train, epoch_acc_train = trainer.train_epoch(epoch)
            train_lost.append(epoch_loss_train)
            train_acc.append(epoch_acc_train)
            epoch_loss_val, epoch_acc_val = trainer.val_epoch(epoch)
            exp_lr_scheduler.step(epoch_loss_val)
            val_lost.append(epoch_loss_val)
            val_acc.append(epoch_acc_val)
    
    print("saving loss and acc history...")
    if operation == constants.OPERATION_TRAINING_FINAL:
        util.saveLearningCurve(os.path.join(path_learning_curves,MODEL_NAME+"-train_lost.txt"), train_lost)
        util.saveLearningCurve(os.path.join(path_learning_curves,MODEL_NAME+"-train_acc.txt"), train_acc)
    elif operation == constants.OPERATION_TRAINING or operation == constants.OPERATION_TRAINING_AUMENTED:
        util.saveLearningCurve(os.path.join(path_learning_curves,MODEL_NAME+"-train_lost.txt"), train_lost)
        util.saveLearningCurve(os.path.join(path_learning_curves,MODEL_NAME+"-train_acc.txt"), train_acc)
        util.saveLearningCurve(os.path.join(path_learning_curves,MODEL_NAME+"-val_lost.txt"), val_lost)
        util.saveLearningCurve(os.path.join(path_learning_curves,MODEL_NAME+"-val_acc.txt"),val_acc)


def __main__():
    # print(train_names)
    # print(train_labels)
    # print(len(train_names), len(test_names))

    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", type=str)
    parser.add_argument("--testModelFile", type=str, default=None)

    parser.add_argument("--pathDataset", type=str, default=constants.PATH_UCFCRIME2LOCAL_FRAMES_REDUCED, help="Directory containing results")
    parser.add_argument("--pathLearningCurves", type=str, default=constants.ANOMALY_PATH_LEARNING_CURVES, help="Directory containing results")
    parser.add_argument("--checkpointPath", type=str, default=constants.ANOMALY_PATH_CHECKPOINTS)
    parser.add_argument("--modelType",type=str,default="alexnet",help="model")
    parser.add_argument("--numEpochs",type=int,default=30)
    parser.add_argument("--batchSize",type=int,default=64)
    parser.add_argument("--featureExtract",type=lambda x: (str(x).lower() == 'true'), default=False, help="to fine tunning")
    parser.add_argument("--schedulerType",type=str,default="OnPlateau",help="learning rate scheduler")
    parser.add_argument("--debuggMode", type=bool, default=False, help="show prints")
    parser.add_argument("--ndis", type=int, help="num dyn imgs")
    parser.add_argument("--joinType", type=str)
    parser.add_argument("--plotSamples", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--shuffle", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--numWorkers", type=int, default=4)
    parser.add_argument("--maxNumFramesOnVideo", type=int, default=0)
    parser.add_argument("--videoSegmentLength", type=int, default=0)
    parser.add_argument("--positionSegment", type=str)
    parser.add_argument("--transferModel", type=str)


    args = parser.parse_args()
    operation = args.operation
    testModelFile = args.testModelFile

    path_dataset = args.pathDataset
    shuffle = args.shuffle
    num_workers = args.numWorkers
    input_size = 224
    maxNumFramesOnVideo = args.maxNumFramesOnVideo
    videoSegmentLength = args.videoSegmentLength
    positionSegment = args.positionSegment
    path_learning_curves = args.pathLearningCurves
    modelType = args.modelType
    batch_size = args.batchSize
    num_epochs = args.numEpochs
    feature_extract = args.featureExtract
    joinType = args.joinType
    scheduler_type = args.schedulerType
    numDiPerVideos = args.ndis
    path_checkpoints = args.checkpointPath
    plot_samples = args.plotSamples
    transferModel = args.transferModel
    # additional_info = 'transferModel: '+transferModel+'_videoSegmentLength-'+str(videoSegmentLength)+'_positionSegment-'+str(positionSegment)
    additional_info = '-'
    transforms = transforms_anomaly.createTransforms(input_size)
    num_classes = 2 #{'Normal_Videos': 0, 'Arrest': 1, 'Assault': 2, 'Burglary': 3, 'Robbery': 4, 'Stealing': 5, 'Vandalism': 6}
    train_videos_path = os.path.join(constants.PATH_UCFCRIME2LOCAL_README, 'Train_split_AD.txt')
    test_videos_path = os.path.join(constants.PATH_UCFCRIME2LOCAL_README, 'Test_split_AD.txt')
    
    # Dalaloaders
    # if operation == constants.OPERATION_TRAINING or operation == constants.OPERATION_TESTING:
        
    # elif operation == constants.OPERATION_TRAINING_FINAL:
        

    #Training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if operation == constants.OPERATION_TRAINING_AUMENTED:
        # path_dataset, batch_size, num_workers, transform, shuffle
        dataloaders_dict = anomalyInitializeDataset.initialize_train_aumented_anomaly_dataset(constants.PATH_DATA_AUMENTATION_OUTPUT,batch_size, 
                                                                                                num_workers,transforms,shuffle)
        additional_info = '-aumented-data'
        training(modelType, num_classes, feature_extract, numDiPerVideos, joinType, device, additional_info, path_learning_curves,
                scheduler_type, num_epochs, dataloaders_dict, path_checkpoints, plot_samples,operation)
    elif operation == constants.OPERATION_TRAINING:
        dataloaders_dict, test_names = anomalyInitializeDataset.initialize_train_val_anomaly_dataset(path_dataset, train_videos_path, test_videos_path,
                                                            batch_size, num_workers,
                                                            numDiPerVideos, transforms, maxNumFramesOnVideo, videoSegmentLength, positionSegment, shuffle)
    elif operation == constants.OPERATION_TRAINING_FINAL:
        dataloaders_dict, test_names = anomalyInitializeDataset.initialize_final_anomaly_dataset(path_dataset, train_videos_path, test_videos_path,
                                                            batch_size, num_workers,
                                                            numDiPerVideos, transforms, maxNumFramesOnVideo, videoSegmentLength, positionSegment, shuffle)
        training(modelType, num_classes, feature_extract, numDiPerVideos, joinType, device, additional_info, path_learning_curves,
                scheduler_type, num_epochs, dataloaders_dict, path_checkpoints, plot_samples,operation)
    elif operation == constants.OPERATION_TESTING:
        dataloaders_dict, test_names = anomalyInitializeDataset.initialize_train_val_anomaly_dataset(path_dataset, train_videos_path, test_videos_path,
                                                            batch_size, num_workers,
                                                            numDiPerVideos, transforms, maxNumFramesOnVideo, videoSegmentLength, positionSegment, shuffle)
        model = torch.load(testModelFile)
        testing(model, dataloaders_dict['test'], device, numDiPerVideos, plot_samples)
    

__main__()






