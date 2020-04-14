import sys
# sys.path.insert(1, '/media/david/datos/PAPERS-SOURCE_CODE/violencedetection')
# sys.path.insert(1, '/Users/davidchoqueluqueroman/Desktop/PAPERS-CODIGOS/violencedetection2')
sys.path.insert(1, '/content/violencedetection2')
import anomalyDataset
import os
import re
from util import video2Images2, saveList, get_model_name
import csv
import pandas as pd
import numpy as np
import torch.nn as nn
import torch
from initializeModel import initialize_model, initializeTransferModel
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
    predictions, scores, gt_labels, test_error, fps = tester.test_model()
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

def training(modelType, transfer_model, num_classes, feature_extract, numDiPerVideos, joinType, device, path_learning_curves, 
                scheduler_type, num_epochs, dataloaders_dict, path_checkpoints, mode, config, learningRate):
    
    if transfer_model is None:
        model, input_size = initialize_model(model_name=modelType, num_classes=num_classes, feature_extract=feature_extract,
                                                numDiPerVideos=numDiPerVideos, 
                                                joinType=joinType, use_pretrained=True)
    else:
        model = initializeTransferModel(model_name=modelType, num_classes=num_classes, feature_extract=feature_extract,
                                                numDiPerVideos=numDiPerVideos, 
                                                joinType=joinType,classifier_file=transfer_model)
    model.to(device)

    # MODEL_NAME = util.get_model_name(modelType, scheduler_type, numDiPerVideos, feature_extract, joinType, num_epochs)
    # MODEL_NAME += additional_info
    # MODEL_NAME = MODEL_NAME+'-FINAL' if operation == constants.OPERATION_TRAINING_FINAL else MODEL_NAME
    # print(MODEL_NAME)
    MODEL_NAME = mode + modelType+'-'+str(numDiPerVideos)+'-Finetuned:'+str(not feature_extract)+'-'+joinType+'-numEpochs:'+str(num_epochs) + config
    params_to_update = verifiParametersToTrain(model, feature_extract)
    optimizer = optim.SGD(params_to_update, lr=learningRate, momentum=0.9)
    # Decay LR by a factor of 0.1 every 7 epochs
    if scheduler_type == "StepLR":
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    elif scheduler_type == "OnPlateau":
        exp_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau( optimizer, patience=5, verbose=True )
    criterion = nn.CrossEntropyLoss()
    
    if mode == 'train':
        trainer = Trainer(model, dataloaders_dict, criterion, optimizer, exp_lr_scheduler, device, num_epochs,
                        os.path.join(path_checkpoints,MODEL_NAME), numDiPerVideos, False, mode, save_model=False)
        train_lost = []
        train_acc = []
        val_lost = []
        val_acc = []
        for epoch in range(1, num_epochs + 1):
            print("----- Epoch {}/{}".format(epoch, num_epochs))
        
            epoch_loss_train, epoch_acc_train = trainer.train_epoch(epoch)
            train_lost.append(epoch_loss_train)
            train_acc.append(epoch_acc_train)
            epoch_loss_val, epoch_acc_val = trainer.val_epoch(epoch)
            exp_lr_scheduler.step(epoch_loss_val)
            val_lost.append(epoch_loss_val)
            val_acc.append(epoch_acc_val)
        
        print("saving loss and acc history...")
        # if operation == constants.OPERATION_TRAINING_FINAL:
        #     util.saveLearningCurve(os.path.join(path_learning_curves,MODEL_NAME+"-train_lost.txt"), train_lost)
        #     util.saveLearningCurve(os.path.join(path_learning_curves,MODEL_NAME+"-train_acc.txt"), train_acc)
        # elif operation == constants.OPERATION_TRAINING or operation == constants.OPERATION_TRAINING_AUMENTED:
        util.saveLearningCurve(os.path.join(path_learning_curves,MODEL_NAME+"-train_lost.txt"), train_lost)
        util.saveLearningCurve(os.path.join(path_learning_curves,MODEL_NAME+"-train_acc.txt"), train_acc)
        util.saveLearningCurve(os.path.join(path_learning_curves,MODEL_NAME+"-val_lost.txt"), val_lost)
        util.saveLearningCurve(os.path.join(path_learning_curves, MODEL_NAME + "-val_acc.txt"), val_acc)
    elif mode == 'test':
        trainer = Trainer(model, dataloaders_dict, criterion, optimizer, exp_lr_scheduler, device, num_epochs,
                        os.path.join(path_checkpoints, MODEL_NAME), numDiPerVideos, False, mode, save_model=True)
        for epoch in range(1, num_epochs + 1):
            print("----- Epoch {}/{}".format(epoch, num_epochs))
        
            epoch_loss_train, epoch_acc_train = trainer.train_epoch(epoch)
            # train_lost.append(epoch_loss_train)
            # train_acc.append(epoch_acc_train)
        
        # auc = testing(model, dataloaders, device, numDiPerVideos, plot_samples)
             
        

import ANOMALYCRIME.datasetUtils as datasetUtils
import ANOMALYCRIME.anomalyDataset as anomalyDataset
import random
import re
from sklearn.model_selection import train_test_split

def remove_short_videos(path, videos, min_num_frames):
    indices2remove = []
    
    for idx, sample in enumerate(videos):
        num_frames = len(os.listdir(os.path.join(path, sample)))
        if num_frames < min_num_frames:
            indices2remove.append(idx)
    
    for idx in sorted(indices2remove, reverse=True):
        del videos[idx]
    
    return videos
         
def train_test_videos(path_train_violence,
                        path_test_violence, 
                        path_train_raw_nonviolence,
                        path_train_new_nonviolence,
                        path_test_raw_nonviolence,
                        path_test_new_nonviolence,
                        proportion_norm_videos,
                        min_num_frames):
    """ load train-test split from original dataset """
    train_names = []
    train_labels = []
    test_names = []
    test_labels = []
    train_bbox_files = []
    test_bbox_files = []

    train_names_violence = util.read_file(path_train_violence)
    train_names_new_nonviolence = util.read_file(path_train_new_nonviolence)
    train_names_raw_nonviolence = util.read_file(path_train_raw_nonviolence)
    test_names_violence = util.read_file(path_test_violence)
    test_names_new_nonviolence = util.read_file(path_test_new_nonviolence)
    test_names_raw_nonviolence = util.read_file(path_test_raw_nonviolence)

    ##Remove normal videos of short duration
    train_names_new_nonviolence = remove_short_videos(constants.PATH_UCFCRIME2LOCAL_FRAMES_NEW_NONVIOLENCE, train_names_new_nonviolence, min_num_frames)
    train_names_raw_nonviolence = remove_short_videos(constants.PATH_UCFCRIME2LOCAL_FRAMES_RAW_NONVIOLENCE, train_names_raw_nonviolence, min_num_frames)
    test_names_new_nonviolence = remove_short_videos(constants.PATH_UCFCRIME2LOCAL_FRAMES_NEW_NONVIOLENCE, test_names_new_nonviolence, min_num_frames)
    test_names_raw_nonviolence = remove_short_videos(constants.PATH_UCFCRIME2LOCAL_FRAMES_RAW_NONVIOLENCE, test_names_raw_nonviolence, min_num_frames)


    ### Train
    # print('Train names: ', len(train_names_violence))
    for tr_name in train_names_violence:
        
        train_names.append(os.path.join(constants.PATH_UCFCRIME2LOCAL_FRAMES_VIOLENCE, tr_name))
        train_labels.append(1)
        video_name = re.findall(r'[\w\.-]+-', tr_name)[0][:-1]
        train_bbox_files.append(os.path.join(constants.PATH_UCFCRIME2LOCAL_BBOX_ANNOTATIONS, video_name+'.txt'))

    ##ramdom normal samples
    negative_samples=[]
    if not os.path.exists(constants.PATH_FINAL_RANDOM_NONVIOLENCE_TRAIN_SPLIT):
        print('Creating Random Normal Train examples file...')
        num_samples = int(2*len(train_names_violence)*proportion_norm_videos)
        train_names_new_nonviolence = random.choices(train_names_new_nonviolence, k=num_samples)
        train_names_raw_nonviolence = random.choices(train_names_raw_nonviolence, k=num_samples)
        for neagtive_name in train_names_new_nonviolence:
            negative_samples.append(os.path.join(constants.PATH_UCFCRIME2LOCAL_FRAMES_NEW_NONVIOLENCE, neagtive_name))
            # train_names.append(os.path.join(constants.PATH_UCFCRIME2LOCAL_FRAMES_NEW_NONVIOLENCE, neagtive_name))
            # train_bbox_files.append(None)
        for neagtive_name in train_names_raw_nonviolence:
            negative_samples.append(os.path.join(constants.PATH_UCFCRIME2LOCAL_FRAMES_RAW_NONVIOLENCE, neagtive_name))
        util.save_file(negative_samples, constants.PATH_FINAL_RANDOM_NONVIOLENCE_TRAIN_SPLIT)
            # train_names.append(os.path.join(constants.PATH_UCFCRIME2LOCAL_FRAMES_RAW_NONVIOLENCE, neagtive_name))
            # train_bbox_files.append(None)
    else:
        negative_samples = util.read_file(constants.PATH_FINAL_RANDOM_NONVIOLENCE_TRAIN_SPLIT)
    
    for sample in negative_samples:
        train_names.append(sample)
        train_bbox_files.append(None)
    negative_labels = [0 for i in range(len(negative_samples))]
    train_labels.extend(negative_labels)
    NumFrames_train = [len(glob.glob1(train_names[i], "*.jpg")) for i in range(len(train_names))]

    ### Test
    for ts_name in test_names_violence:
        test_names.append(os.path.join(constants.PATH_UCFCRIME2LOCAL_FRAMES_VIOLENCE, ts_name))
        test_labels.append(1)
        video_name = re.findall(r'[\w\.-]+-', ts_name)[0][:-1]
        test_bbox_files.append(os.path.join(constants.PATH_UCFCRIME2LOCAL_BBOX_ANNOTATIONS, video_name+'.txt'))
   
    negative_samples=[]
    if not os.path.exists(constants.PATH_FINAL_RANDOM_NONVIOLENCE_TEST_SPLIT):
        print('Creating Random Normal Test examples file...')
        num_samples = int(2*len(test_names_violence)*proportion_norm_videos)
        test_names_new_nonviolence = random.choices(test_names_new_nonviolence, k=num_samples)
        test_names_raw_nonviolence = random.choices(test_names_raw_nonviolence, k=num_samples)
        for neagtive_name in test_names_new_nonviolence:
            negative_samples.append(os.path.join(constants.PATH_UCFCRIME2LOCAL_FRAMES_NEW_NONVIOLENCE, neagtive_name))
            
        for neagtive_name in test_names_raw_nonviolence:
            negative_samples.append(os.path.join(constants.PATH_UCFCRIME2LOCAL_FRAMES_RAW_NONVIOLENCE, neagtive_name))
        util.save_file(negative_samples, constants.PATH_FINAL_RANDOM_NONVIOLENCE_TEST_SPLIT)
           
    else:
        negative_samples = util.read_file(constants.PATH_FINAL_RANDOM_NONVIOLENCE_TEST_SPLIT)
    for sample in negative_samples:
        test_names.append(sample)
        test_bbox_files.append(None)
    negative_labels = [0 for i in range(len(negative_samples))]
    test_labels.extend(negative_labels)
    NumFrames_test = [len(glob.glob1(test_names[i], "*.jpg")) for i in range(len(test_names))]


    print(len(train_names), len(train_labels), len(NumFrames_train), len(train_bbox_files), len(test_names), len(test_labels), len(NumFrames_test), len(test_bbox_files))
    data = {'train_names':train_names,
            'train_labels':train_labels,
            'NumFrames_train':NumFrames_train,
            'train_bbox_files':train_bbox_files,
            'test_names':test_names,
            'test_labels':test_labels,
            'NumFrames_test':NumFrames_test,
            'test_bbox_files':test_bbox_files}
    return data

def check_data(path_example, numFrames, label, tmpAnnotation):
    num_frames_real = os.listdir(path_example)
    if len(num_frames_real) != numFrames:
        print('Error in num Frames:{}, real: {}, data: {}'.format(path_example, len(num_frames_real), numFrames))
    if label == 0 and tmpAnnotation is not None:
        print('Error in temporal annotation:{}, label: {}, data: {}'.format(path_example, label, tmpAnnotation))

def initialize_train_val_anomaly_dataset(batch_size,
                                        num_workers,
                                        numDiPerVideos,
                                        transforms,
                                        videoSegmentLength,
                                        positionSegment,
                                        shuffle,
                                        overlaping,
                                        frame_skip):

    train_violence = constants.PATHS_SPLITS_DICT['train_violence']
    test_violence = constants.PATHS_SPLITS_DICT['test_violence']
    train_raw_nonviolence = constants.PATHS_SPLITS_DICT['train_raw_nonviolence']
    train_new_nonviolence = constants.PATHS_SPLITS_DICT['train_new_nonviolence']
    test_raw_nonviolence = constants.PATHS_SPLITS_DICT['test_raw_nonviolence']
    test_new_nonviolence = constants.PATHS_SPLITS_DICT['test_new_nonviolence']


    data = train_test_videos(path_train_violence=train_violence,
                                path_test_violence=test_violence, 
                                path_train_raw_nonviolence=train_raw_nonviolence,
                                path_train_new_nonviolence=train_new_nonviolence,
                                path_test_raw_nonviolence=test_raw_nonviolence,
                                path_test_new_nonviolence=test_new_nonviolence,
                                proportion_norm_videos=0.5,
                                min_num_frames = 50)
    train_names = data['train_names']
    train_labels = data['train_labels']
    train_num_frames = data['NumFrames_train']
    train_bbox_files = data['train_bbox_files'] 
    test_names = data['test_names']
    test_labels = data['test_labels']
    test_num_frames = data['NumFrames_test']
    test_bbox_files = data['test_bbox_files']

    combined = list(zip(train_names, train_num_frames, train_bbox_files))
    combined_train_names, combined_val_names, train_labels, val_labels = train_test_split(combined, train_labels, stratify=train_labels, test_size=0.30, shuffle=True)
    train_names, train_num_frames, train_bbox_files = zip(*combined_train_names)
    val_names, val_num_frames, val_bbox_files = zip(*combined_val_names)

    # print('validation set:', len(val_names), len(val_labels), len(val_num_frames), len(val_bbox_files))
    util.print_balance(train_labels, 'train')
    util.print_balance(val_labels, 'val')
    util.print_balance(test_labels, 'test')

    # print('Train sanity check')
    # for idx,dt in enumerate(train_names):
    #     check_data(dt,train_num_frames[idx],train_labels[idx], train_bbox_files[idx])

    # print('Train sanity check')
    # for idx,dt in enumerate(test_names):
    #     check_data(dt, test_num_frames[idx],test_labels[idx], test_bbox_files[idx])

    # print('Train sanity check')
    # for idx,dt in enumerate(val_names):
    #     check_data(dt, val_num_frames[idx],val_labels[idx], val_bbox_files[idx])

    dataloaders_dict = None
    image_datasets = {
        "train": anomalyDataset.AnomalyDataset(dataset=train_names,
                                                labels=train_labels,
                                                numFrames=train_num_frames,
                                                bbox_files=train_bbox_files,
                                                spatial_transform=transforms["train"],
                                                nDynamicImages=numDiPerVideos,
                                                videoSegmentLength=videoSegmentLength,
                                                positionSegment=positionSegment,
                                                overlaping=overlaping,
                                                frame_skip=frame_skip),
        "val": anomalyDataset.AnomalyDataset(dataset=val_names,
                                                labels=val_labels,
                                                numFrames=val_num_frames,
                                                bbox_files=val_bbox_files,
                                                spatial_transform=transforms["val"], 
                                                nDynamicImages=numDiPerVideos,
                                                videoSegmentLength=videoSegmentLength,
                                                positionSegment=positionSegment,
                                                overlaping=overlaping,
                                                frame_skip=frame_skip),
        "test": anomalyDataset.AnomalyDataset(dataset=test_names,
                                                labels=test_labels,
                                                numFrames=test_num_frames,
                                                bbox_files=test_bbox_files,
                                                spatial_transform=transforms["test"],
                                                nDynamicImages=numDiPerVideos,
                                                videoSegmentLength=videoSegmentLength,
                                                positionSegment=positionSegment,
                                                overlaping=overlaping,
                                                frame_skip=frame_skip)
    }
    dataloaders_dict = {
        "train": torch.utils.data.DataLoader(image_datasets["train"], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers),
        "val": torch.utils.data.DataLoader( image_datasets["val"], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers),
        "test": torch.utils.data.DataLoader( image_datasets["test"], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    }
     
    return dataloaders_dict

def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--modelType",type=str,default="alexnet",help="model")
    parser.add_argument("--numEpochs",type=int,default=30)
    parser.add_argument("--batchSize",type=int,default=64)
    parser.add_argument("--featureExtract",type=lambda x: (str(x).lower() == 'true'), default=False, help="to fine tunning")
    parser.add_argument("--debuggMode", type=bool, default=False, help="show prints")
    parser.add_argument("--ndis", type=int, help="num dyn imgs")
    parser.add_argument("--joinType", type=str)
    parser.add_argument("--overlaping", type=float)
    parser.add_argument("--learningRate", type=float)
    parser.add_argument("--shuffle", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--numWorkers", type=int, default=4)
    parser.add_argument("--videoSegmentLength", type=int, default=0)
    parser.add_argument("--positionSegment", type=str)
    parser.add_argument("--frame_skip", type=int)
    


    args = parser.parse_args()
    mode = args.mode
    # path_dataset = constants.PATH_UCFCRIME2LOCAL_FRAMES
    path_learning_curves = constants.ANOMALY_PATH_LEARNING_CURVES
    path_checkpoints = constants.ANOMALY_PATH_CHECKPOINTS
    # train_videos_path = os.path.join(constants.PATH_UCFCRIME2LOCAL_README, 'Train_split_AD.txt')
    # test_videos_path = os.path.join(constants.PATH_UCFCRIME2LOCAL_README, 'Test_split_AD.txt')

    shuffle = args.shuffle
    num_workers = args.numWorkers
    input_size = 224
    videoSegmentLength = args.videoSegmentLength
    positionSegment = args.positionSegment
    modelType = args.modelType
    batch_size = args.batchSize
    num_epochs = args.numEpochs
    feature_extract = args.featureExtract
    joinType = args.joinType
    scheduler_type = 'OnPlateau'
    numDiPerVideos = args.ndis
    overlaping = args.overlaping
    learningRate = args.learningRate
    frame_skip = args.frame_skip
    
    transforms = transforms_anomaly.createTransforms(input_size)
    num_classes = 2 
    #Training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # if operation == constants.OPERATION_TRAINING_TRANSFER:
    #     # path_dataset, batch_size, num_workers, transform, shuffle
    #     dataloaders_dict = anomalyInitializeDataset.initialize_train_aumented_anomaly_dataset(constants.PATH_DATA_AUMENTATION_OUTPUT,batch_size, 
    #                                                                                             num_workers,transforms,shuffle)
    #     additional_info = '-transfered-data'
    #     transferLearning(modelType, num_classes, feature_extract, numDiPerVideos, joinType, device, additional_info, path_learning_curves,
    #             scheduler_type, num_epochs, dataloaders_dict, path_checkpoints, plot_samples,operation)
    # elif operation == constants.OPERATION_TRAINING_AUMENTED:
    #     # path_dataset, batch_size, num_workers, transform, shuffle
    #     dataloaders_dict = anomalyInitializeDataset.initialize_train_aumented_anomaly_dataset(constants.PATH_DATA_AUMENTATION_OUTPUT,batch_size, 
    #                                                                                             num_workers,transforms,shuffle, val_split=0.35)
    #     additional_info = '-aumented-data-30'
    #     feature_extract = False
    #     training(modelType, num_classes, feature_extract, numDiPerVideos, joinType, device, additional_info, path_learning_curves,
    #             scheduler_type, num_epochs, dataloaders_dict, path_checkpoints, plot_samples,operation)
    only_violence = True
    config = '-videoSegmentLength:'+str(videoSegmentLength)+'-overlaping:'+str(overlaping)+'-only_violence:'+str(only_violence)+'-skipFrame:'+str(frame_skip)
    if mode == 'train':

        dataloaders_dict = initialize_train_val_anomaly_dataset(batch_size,
                                                                num_workers,
                                                                numDiPerVideos,
                                                                transforms,
                                                                videoSegmentLength,
                                                                positionSegment,
                                                                shuffle,
                                                                overlaping=overlaping,
                                                                frame_skip=frame_skip)

        training(modelType=modelType, transfer_model=None, num_classes=num_classes, feature_extract=feature_extract, numDiPerVideos=numDiPerVideos,
                joinType=joinType, device=device, path_learning_curves=path_learning_curves, scheduler_type=scheduler_type,
                num_epochs=num_epochs, dataloaders_dict=dataloaders_dict, path_checkpoints=path_checkpoints, mode=mode, config=config,
                learningRate=learningRate) 
    # elif mode == 'test':
    #     dataloaders_dict, test_names = anomalyInitializeDataset.initialize_train_test_anomaly_dataset(path_dataset, train_videos_path,
    #                                                     test_videos_path, batch_size, num_workers, numDiPerVideos, transforms,
    #                                                     videoSegmentLength, positionSegment, shuffle, only_violence=only_violence, overlaping=overlaping)
    #     training(modelType, None, num_classes, feature_extract, numDiPerVideos, joinType, device, path_learning_curves,
    #             scheduler_type, num_epochs, dataloaders_dict, path_checkpoints,mode, config, learningRate)
  
    

__main__()






