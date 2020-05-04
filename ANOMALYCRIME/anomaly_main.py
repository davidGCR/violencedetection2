import sys
# sys.path.insert(1, '/media/david/datos/PAPERS-SOURCE_CODE/violencedetection')
sys.path.insert(1, '/Users/davidchoqueluqueroman/Desktop/PAPERS-CODIGOS/violencedetection2')
# sys.path.insert(1, '/content/violencedetection2')
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
import torchvision.transforms as transforms

import torchvision
from torch.utils.tensorboard import SummaryWriter
from kfolds import *
from operator import itemgetter
import copy

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

# helper function to show an image
# (used in the `plot_classes_preds` function below)
def min_max_normalize_tensor(img):
    # print("normalize:", img.size())
    _min = torch.min(img)
    _max = torch.max(img)
    # print("min:", _min.item(), ", max:", _max.item())
    return (img - _min) / (_max - _min)

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    
    # img = img / 2 + 0.5     # unnormalize
    # for im in img[0]:
    # min_max_normalize_tensor(img) 
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def computeNormalizationValues(dataloader):
    pop_mean = []
    pop_std0 = []
    pop_std1 = []
    for i,data in enumerate(dataloader, 0):
        image, label, vid_name, _ = data # shape (batch_size, 3, height, width)
        numpy_image = image.numpy()
        batch_mean = np.mean(numpy_image, axis=(0,2,3))
        batch_std0 = np.std(numpy_image, axis=(0,2,3))
        batch_std1 = np.std(numpy_image, axis=(0,2,3), ddof=1)
        pop_mean.append(batch_mean)
        pop_std0.append(batch_std0)
        pop_std1.append(batch_std1)
    
    pop_mean = np.array(pop_mean).mean(axis=0)
    pop_std0 = np.array(pop_std0).mean(axis=0)
    pop_std1 = np.array(pop_std1).mean(axis=0)
    return pop_mean, pop_std0, pop_std1

def meanStdDataset():
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

    # combined = list(zip(train_names, train_num_frames, train_bbox_files))
    # combined_train_names, combined_val_names, train_labels, val_labels = train_test_split(combined, train_labels, stratify=train_labels, test_size=0.30, shuffle=True)
    # train_names, train_num_frames, train_bbox_files = zip(*combined_train_names)
    # val_names, val_num_frames, val_bbox_files = zip(*combined_val_names)
    
    transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.ToTensor()
    ])
    train_dataset = anomalyDataset.AnomalyDataset(dataset=train_names,
                                                labels=train_labels,
                                                numFrames=train_num_frames,
                                                bbox_files=train_bbox_files,
                                                spatial_transform=transform,
                                                nDynamicImages=1,
                                                videoSegmentLength=10,
                                                positionSegment='begin',
                                                overlaping=0,
                                                frame_skip=10)
    
    # validation_dataset = anomalyDataset.AnomalyDataset(dataset=val_names,
    #                                             labels=val_labels,
    #                                             numFrames=val_num_frames,
    #                                             bbox_files=val_bbox_files,
    #                                             spatial_transform=transform,
    #                                             nDynamicImages=1,
    #                                             videoSegmentLength=10,
    #                                             positionSegment='begin',
    #                                             overlaping=0,
    #                                             frame_skip=20)                                                
    dataloader_train = torch.utils.data.DataLoader(train_dataset, batch_size=4096, shuffle=False, num_workers=4)
    # dataloader_val = torch.utils.data.DataLoader(validation_dataset, batch_size=4096, shuffle=False, num_workers=4)

    pop_mean, pop_std0, pop_std1 = computeNormalizationValues(dataloader_train)
    print('Train Dataset: ', pop_mean, pop_std0, pop_std1)
    # pop_mean, pop_std0, pop_std1 = computeNormalizationValues(dataloader_val)
    # print('Validation Dataset: ', pop_mean, pop_std0, pop_std1)
    


def training(model, numDiPerVideos, criterion, optimizer, scheduler, device, num_epochs, dataloaders_dict, path_checkpoints, mode, board_folder, split_type):
    if split_type == 'train-test':
        train_dataloader = dataloaders_dict['train']
        val_dataloader = dataloaders_dict['test']
    elif split_type == 'train-val-test':
        train_dataloader = dataloaders_dict['train']
        val_dataloader = dataloaders_dict['val']
   
    writer = SummaryWriter('runs/'+board_folder)
    trainer = Trainer(model=model,
                        train_dataloader=train_dataloader,
                        val_dataloader=val_dataloader,
                        criterion=criterion,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        device=device,
                        num_epochs=num_epochs,
                        checkpoint_path=path_checkpoints,
                        numDynamicImage=numDiPerVideos,
                        plot_samples=False,
                        train_type=mode,
                        save_model=False)
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
        scheduler.step()
        val_lost.append(epoch_loss_val)
        val_acc.append(epoch_acc_val)
        writer.add_scalar('training loss', epoch_loss_train, epoch)
        writer.add_scalar('validation loss', epoch_loss_val, epoch)
        writer.add_scalar('training Acc', epoch_acc_train, epoch)
        writer.add_scalar('validation Acc', epoch_acc_val, epoch)
               
        

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

    new_split = False
    ### Train
    # print('Train names: ', len(train_names_violence))
    for tr_name in train_names_violence:
        train_names.append(os.path.join(constants.PATH_UCFCRIME2LOCAL_FRAMES_VIOLENCE, tr_name))
        train_labels.append(1)
        video_name = re.findall(r'[\w\.-]+-', tr_name)[0][:-1]
        train_bbox_files.append(os.path.join(constants.PATH_VIOLENCECRIME2LOCAL_BBOX_ANNOTATIONS, video_name+'.txt'))

    ##ramdom normal samples
    negative_samples=[]
    if not os.path.exists(constants.PATH_FINAL_RANDOM_NONVIOLENCE_TRAIN_SPLIT):
        print('Creating Random Normal Train examples file...')
        num_new_samples = int(len(train_names)*proportion_norm_videos)
        train_names_new_nonviolence = random.choices(train_names_new_nonviolence, k=num_new_samples)
        train_names_raw_nonviolence = random.choices(train_names_raw_nonviolence, k=len(train_names) - num_new_samples)
        if len(train_names_new_nonviolence) == 0:
            print('Using only raw non violence videos...')
        for neagtive_name in train_names_new_nonviolence:
            negative_samples.append(os.path.join(constants.PATH_UCFCRIME2LOCAL_FRAMES_NEW_NONVIOLENCE, neagtive_name))
            # train_names.append(os.path.join(constants.PATH_UCFCRIME2LOCAL_FRAMES_NEW_NONVIOLENCE, neagtive_name))
            # train_bbox_files.append(None)
        for neagtive_name in train_names_raw_nonviolence:
            negative_samples.append(os.path.join(constants.PATH_UCFCRIME2LOCAL_FRAMES_RAW_NONVIOLENCE, neagtive_name))
        util.save_file(negative_samples, constants.PATH_FINAL_RANDOM_NONVIOLENCE_TRAIN_SPLIT)
            # train_names.append(os.path.join(constants.PATH_UCFCRIME2LOCAL_FRAMES_RAW_NONVIOLENCE, neagtive_name))
            # train_bbox_files.append(None)
        new_split = True
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
        test_bbox_files.append(os.path.join(constants.PATH_VIOLENCECRIME2LOCAL_BBOX_ANNOTATIONS, video_name+'.txt'))
   
    negative_samples=[]
    if not os.path.exists(constants.PATH_FINAL_RANDOM_NONVIOLENCE_TEST_SPLIT):
        print('Creating Random Normal Test examples file...')
        num_samples = int(len(test_names)*proportion_norm_videos)
        test_names_new_nonviolence = random.choices(test_names_new_nonviolence, k=num_samples)
        test_names_raw_nonviolence = random.choices(test_names_raw_nonviolence, k=len(test_names) - num_new_samples)
        for neagtive_name in test_names_new_nonviolence:
            negative_samples.append(os.path.join(constants.PATH_UCFCRIME2LOCAL_FRAMES_NEW_NONVIOLENCE, neagtive_name))
            
        for neagtive_name in test_names_raw_nonviolence:
            negative_samples.append(os.path.join(constants.PATH_UCFCRIME2LOCAL_FRAMES_RAW_NONVIOLENCE, neagtive_name))
        util.save_file(negative_samples, constants.PATH_FINAL_RANDOM_NONVIOLENCE_TEST_SPLIT)
        new_split = True
    else:
        negative_samples = util.read_file(constants.PATH_FINAL_RANDOM_NONVIOLENCE_TEST_SPLIT)
    for sample in negative_samples:
        test_names.append(sample)
        test_bbox_files.append(None)
    negative_labels = [0 for i in range(len(negative_samples))]
    test_labels.extend(negative_labels)
    NumFrames_test = [len(glob.glob1(test_names[i], "*.jpg")) for i in range(len(test_names))]


    print('Train Split: ', len(train_names), len(train_labels), len(NumFrames_train), len(train_bbox_files), ', Test Split: ',len(test_names), len(test_labels), len(NumFrames_test), len(test_bbox_files))
    data = {'train_names':train_names,
            'train_labels':train_labels,
            'NumFrames_train':NumFrames_train,
            'train_bbox_files':train_bbox_files,
            'test_names':test_names,
            'test_labels':test_labels,
            'NumFrames_test':NumFrames_test,
            'test_bbox_files':test_bbox_files}
    return data, new_split

def check_data(path_example, numFrames, label, tmpAnnotation):
    num_frames_real = os.listdir(path_example)
    if len(num_frames_real) != numFrames:
        print('Error in num Frames:{}, real: {}, data: {}'.format(path_example, len(num_frames_real), numFrames))
    if label == 0 and tmpAnnotation is not None:
        print('Error in temporal annotation:{}, label: {}, data: {}'.format(path_example, label, tmpAnnotation))


def initialize_dataloaders(batch_size,
                            num_workers,
                            numDiPerVideos,
                            transforms,
                            videoSegmentLength,
                            positionSegment,
                            shuffle,
                            overlaping,
                            frame_skip,
                            split_type,
                            folds_number=1):

    train_violence = constants.PATHS_SPLITS_DICT['train_violence']
    test_violence = constants.PATHS_SPLITS_DICT['test_violence']
    train_raw_nonviolence = constants.PATHS_SPLITS_DICT['train_raw_nonviolence']
    train_new_nonviolence = constants.PATHS_SPLITS_DICT['train_new_nonviolence']
    test_raw_nonviolence = constants.PATHS_SPLITS_DICT['test_raw_nonviolence']
    test_new_nonviolence = constants.PATHS_SPLITS_DICT['test_new_nonviolence']


    data, new_split = train_test_videos(path_train_violence=train_violence,
                                path_test_violence=test_violence, 
                                path_train_raw_nonviolence=train_raw_nonviolence,
                                path_train_new_nonviolence=train_new_nonviolence,
                                path_test_raw_nonviolence=test_raw_nonviolence,
                                path_test_new_nonviolence=test_new_nonviolence,
                                proportion_norm_videos=0,
                                min_num_frames = 50)
    train_names = data['train_names']
    train_labels = data['train_labels']
    train_num_frames = data['NumFrames_train']
    train_bbox_files = data['train_bbox_files'] 
    test_names = data['test_names']
    test_labels = data['test_labels']
    test_num_frames = data['NumFrames_test']
    test_bbox_files = data['test_bbox_files']

    dataloaders_dict = None
    if split_type == 'train-val-test':
        combined = list(zip(train_names, train_num_frames, train_bbox_files))
        combined_train_names, combined_val_names, train_labels, val_labels = train_test_split(combined, train_labels, stratify=train_labels, test_size=0.25, shuffle=True)
        train_names, train_num_frames, train_bbox_files = zip(*combined_train_names)
        val_names, val_num_frames, val_bbox_files = zip(*combined_val_names)
        util.print_balance(train_labels, 'train')
        util.print_balance(val_labels, 'val')
        util.print_balance(test_labels, 'test')
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
            "val": torch.utils.data.DataLoader( image_datasets["val"], batch_size=batch_size, shuffle=False, num_workers=num_workers),
            "test": torch.utils.data.DataLoader( image_datasets["test"], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        }

    elif split_type == 'train-test':
        util.print_balance(train_labels, 'train')
        util.print_balance(test_labels, 'test')
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
            "test": torch.utils.data.DataLoader( image_datasets["test"], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        }
    elif split_type == 'cross-val':
        folds_number = 5
        # train_names = train_names.extend(test_names)
        # train_labels = train_labels.extend(test_labels)
        # train_num_frames = train_num_frames.extend(test_num_frames)
        # train_bbox_files = train_bbox_files.extend(test_bbox_files)
        train_names = train_names + test_names
        train_labels = train_labels + test_labels
        train_num_frames = train_num_frames + test_num_frames
        train_bbox_files = train_bbox_files + test_bbox_files

        train_all = list(zip(train_names, train_labels, train_num_frames, train_bbox_files))
        random.shuffle(train_all)
        names, labels, num_frames, bbox_files = zip(*train_all)
        dataloaders_list_splits = []
        split=0
        for train_idx, test_idx in k_folds(n_splits=folds_number, subjects=len(train_names)):
            split +=1
            train_x = list(itemgetter(*train_idx)(names))
            train_y = list(itemgetter(*train_idx)(labels))
            train_num_frames = list(itemgetter(*train_idx)(num_frames))
            train_bbox_files = list(itemgetter(*train_idx)(bbox_files))

            test_x = list(itemgetter(*test_idx)(names))
            test_y = list(itemgetter(*test_idx)(labels))
            test_num_frames = list(itemgetter(*test_idx)(num_frames))
            test_bbox_files = list(itemgetter(*test_idx)(bbox_files))
            print('K-fold-Split: {}'.format(split))
            util.print_balance(train_y, 'train')
            util.print_balance(test_y, 'test')
            image_datasets = {
                "train": anomalyDataset.AnomalyDataset(dataset=train_x,
                                                        labels=train_y,
                                                        numFrames=train_num_frames,
                                                        bbox_files=train_bbox_files,
                                                        spatial_transform=transforms["train"],
                                                        nDynamicImages=numDiPerVideos,
                                                        videoSegmentLength=videoSegmentLength,
                                                        positionSegment=positionSegment,
                                                        overlaping=overlaping,
                                                        frame_skip=frame_skip),
                "test": anomalyDataset.AnomalyDataset(dataset=test_x,
                                                        labels=test_y,
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
                "test": torch.utils.data.DataLoader( image_datasets["test"], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
            }
            dataloaders_list_splits.append(dataloaders_dict)
            dataloaders_dict = dataloaders_list_splits
    return dataloaders_dict, new_split

import SALIENCY.saliencyModel
from SALIENCY.loss import Loss
from tqdm import tqdm
from torch.autograd import Variable
import include

def train_mask_model(num_epochs, regularizers, device, checkpoint_path, dataloaders_dict, black_box_file, numDynamicImages, image_idx):
    num_classes = 2
    saliency_m = SALIENCY.saliencyModel.build_saliency_model(num_classes=num_classes)
    saliency_m = saliency_m.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(saliency_m.parameters())           
    black_box_model = torch.load(black_box_file)
    black_box_model = black_box_model.to(device)
    black_box_model.inferenceMode(numDynamicImages)

    loss_func = Loss(num_classes=num_classes, regularizers=regularizers, num_dynamic_images=numDynamicImages)
    best_loss = 1000.0
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        print("----- Epoch {}/{}".format(epoch+1, num_epochs))
        running_loss = 0.0
        running_loss_train = 0.0

        for i, data in tqdm(enumerate(dataloaders_dict['train'], 0)):
            # get the inputs
            inputs, labels, video_name, bbox_segments = data #dataset load [bs,ndi,c,w,h]
            # print('dataset element: ',inputs.shape) #torch.Size([8, 1, 3, 224, 224])
            batch_size = inputs.size()[0]
            # print(batch_size)
            if numDynamicImages > 1:
                inputs = inputs.permute(1, 0, 2, 3, 4) #[ndi,bs,c,w,h]
            # print('inputs shape:',inputs.shape)
            # wrap them in Variable
            inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device))
            # zero the parameter gradients
            optimizer.zero_grad()
            mask, out = saliency_m(inputs[image_idx], labels)
            # print('mask shape:', mask.shape)
            # print('inputs shape:',inputs.shape)
            # print('labels shape:', labels.shape)
            # print(labels)
            # inputs_r = Variable(inputs_r.cuda())
            loss = loss_func.get(mask,inputs,labels,black_box_model)
            # running_loss += loss.data[0]
            running_loss += loss.item()
            running_loss_train += loss.item()*batch_size
            # if(i%10 == 0):
            #     print('Epoch = %f , Loss = %f '%(epoch+1 , running_loss/(batch_size*(i+1))) )
            loss.backward()
            optimizer.step()
        # exp_lr_scheduler.step(running_loss)

        epoch_loss = running_loss / len(dataloaders_dict["train"].dataset)
        epoch_loss_train = running_loss_train / len(dataloaders_dict["train"].dataset)
        print("{} RawLoss: {:.4f} Loss: {:.4f}".format('train', epoch_loss, epoch_loss_train))

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            # self.best_model_wts = copy.deepcopy(self.model.state_dict())
            print('Saving entire saliency model...', checkpoint_path)
            torch.save(saliency_m, checkpoint_path) 
            # save_checkpoint(saliency_m, checkpoint_path)



def __main_mask__():
    parser = argparse.ArgumentParser()
    parser.add_argument("--numEpochs",type=int,default=30)
    parser.add_argument("--batchSize", type=int, default=64)
    parser.add_argument("--numWorkers", type=int, default=4)
    parser.add_argument("--ndis", type=int, help="num dyn imgs")
    parser.add_argument("--videoSegmentLength", type=int, default=0)
    parser.add_argument("--positionSegment", type=str)
    parser.add_argument("--shuffle", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--frame_skip", type=int)
    parser.add_argument("--overlaping", type=float)
    parser.add_argument("--split_type", type=str)
    parser.add_argument("--image_idx", type=int)

    parser.add_argument("--areaL", type=float, default=None)
    parser.add_argument("--smoothL", type=float, default=None)
    parser.add_argument("--preserverL", type=float, default=None)
    parser.add_argument("--areaPowerL", type=float, default=None)
    args = parser.parse_args()
    if args.areaL == None:
        areaL = 8
    else:
        areaL = args.areaL
        checkpoint_info += '_areaL-'+str(args.areaL)
    if args.smoothL == None:
        smoothL = 0.5
    else:
        smoothL = args.smoothL
        checkpoint_info += '_smoothL-' + str(args.smoothL)
    if args.preserverL == None:
        preserverL = 0.3
    else:
        preserverL = args.preserverL
        checkpoint_info += '_preserverL-' + str(args.preserverL)
    if args.areaPowerL == None:
        areaPowerL = 0.3
    else:
        areaPowerL = args.areaPowerL
    regularizers = {'area_loss_coef': areaL,
                    'smoothness_loss_coef': smoothL,
                    'preserver_loss_coef': preserverL,
                    'area_loss_power': areaPowerL}
    
    input_size = 224
    transforms = transforms_anomaly.createTransforms(input_size)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataloaders_dict, new_split = initialize_dataloaders(args.batchSize,
                                                            args.numWorkers,
                                                            args.ndis,
                                                            transforms,
                                                            args.videoSegmentLength,
                                                            args.positionSegment,
                                                            args.shuffle,
                                                            overlaping=args.overlaping,
                                                            frame_skip=args.frame_skip,
                                                            split_type=args.split_type)
    
    experimentConfig = 'Mask_model, segmentLen-{}, numDynIms-{}, frameSkip-{}, epochs-{}, split_type-{}, image_idx-{}'.format(
                                                                                                args.videoSegmentLength,
                                                                                                args.ndis,
                                                                                                args.frame_skip,
                                                                                                args.numEpochs,
                                                                                                args.split_type,
                                                                                                args.image_idx)
    path_checkpoints = os.path.join(constants.ANOMALY_PATH_CHECKPOINTS, experimentConfig)
    black_box_file = include.root+'/ANOMALY_RESULTS/checkpoints/Model-resnet18, segmentLen-20, numDynIms-6, frameSkip-0, epochs-10, new_split-False, split_type-train-test'

    train_mask_model(args.numEpochs, regularizers, device, path_checkpoints, dataloaders_dict, black_box_file, args.ndis, args.image_idx)

def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--pathLearningCurves", type=str, default=constants.ANOMALY_PATH_LEARNING_CURVES)
    # parser.add_argument("--pathCheckpoints", type=str, default=constants.ANOMALY_PATH_CHECKPOINTS)
    parser.add_argument("--modelType",type=str,default="alexnet",help="model")
    parser.add_argument("--numEpochs",type=int,default=30)
    parser.add_argument("--batchSize",type=int,default=64)
    parser.add_argument("--featureExtract",type=lambda x: (str(x).lower() == 'true'), default=False, help="to fine tunning")
    parser.add_argument("--debuggMode", type=bool, default=False, help="show prints")
    parser.add_argument("--ndis", type=int, help="num dyn imgs")
    parser.add_argument("--joinType", type=str)
    parser.add_argument("--overlaping", type=float)
    parser.add_argument("--learningRate", type=float)
    parser.add_argument("--scheduler", type=str)
    parser.add_argument("--shuffle", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--numWorkers", type=int, default=4)
    parser.add_argument("--videoSegmentLength", type=int, default=0)
    parser.add_argument("--positionSegment", type=str)
    parser.add_argument("--frame_skip", type=int)
    parser.add_argument("--boardFolder", type=str)
    parser.add_argument("--split_type", type=str)


    args = parser.parse_args()
    mode = args.mode
    # path_dataset = constants.PATH_UCFCRIME2LOCAL_FRAMES
    path_learning_curves = args.pathLearningCurves
    
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
    scheduler_type = args.scheduler
    numDiPerVideos = args.ndis
    overlaping = args.overlaping
    learningRate = args.learningRate
    frame_skip = args.frame_skip
    board_folder = args.boardFolder
    transforms = transforms_anomaly.createTransforms(input_size)
    num_classes = 2
    split_type = args.split_type
    #Training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    only_violence = True

    
    # '-videoSegmentLength:'+str(videoSegmentLength)+'-overlaping:'+str(overlaping)+'-only_violence:'+str(only_violence)+'-skipFrame:'+str(frame_skip)

    dataloaders_dict, new_split = initialize_dataloaders(batch_size,
                                                            num_workers,
                                                            numDiPerVideos,
                                                            transforms,
                                                            videoSegmentLength,
                                                            positionSegment,
                                                            shuffle,
                                                            overlaping=overlaping,
                                                            frame_skip=frame_skip,
                                                            split_type=split_type)
    
    experimentConfig = 'Model-{}, segmentLen-{}, numDynIms-{}, frameSkip-{}, epochs-{}, new_split-{}, split_type-{}'.format(modelType,
                                                                                                videoSegmentLength,
                                                                                                numDiPerVideos,
                                                                                                frame_skip,
                                                                                                num_epochs,
                                                                                                new_split,
                                                                                                split_type)
    path_checkpoints = os.path.join(constants.ANOMALY_PATH_CHECKPOINTS, experimentConfig)

    if split_type == 'cross-val':
        models = []
        for split, dlt in enumerate(dataloaders_dict):
            print('====== Fold {}'.format(split+1))
            model, input_size = initialize_model(model_name=modelType, num_classes=num_classes, feature_extract=feature_extract,
                                                numDiPerVideos=numDiPerVideos, 
                                                joinType=joinType, use_pretrained=True)
            model = model.to(device)
            params_to_update = verifiParametersToTrain(model, feature_extract)
            optimizer = optim.SGD(params_to_update, lr=learningRate, momentum=0.9)
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
            criterion = nn.CrossEntropyLoss()
            
            training(model=model, numDiPerVideos=numDiPerVideos,criterion=criterion, optimizer=optimizer, scheduler=exp_lr_scheduler, device=device,
                num_epochs=num_epochs, dataloaders_dict=dlt, path_checkpoints=path_checkpoints, mode=mode, board_folder=experimentConfig+'-'+str(split+1),
                split_type='train-test') 

    else:
        
        model, input_size = initialize_model(model_name=modelType, num_classes=num_classes, feature_extract=feature_extract,
                                                numDiPerVideos=numDiPerVideos, 
                                                joinType=joinType, use_pretrained=True)
        model = model.to(device)
        params_to_update = verifiParametersToTrain(model, feature_extract)
        optimizer = optim.SGD(params_to_update, lr=learningRate, momentum=0.9)
        # Decay LR by a factor of 0.1 every 7 epochs
        # if scheduler_type == "StepLR":
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        # elif scheduler_type == "OnPlateau":
        #     exp_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau( optimizer, patience=5, verbose=True )
        criterion = nn.CrossEntropyLoss()
        # meanStdDataset()
        # model, numDiPerVideos, criterion, optimizer, scheduler, device, num_epochs, dataloaders_dict, path_checkpoints, mode, board_folder
        training(model=model, numDiPerVideos=numDiPerVideos,criterion=criterion, optimizer=optimizer, scheduler=exp_lr_scheduler, device=device,
                num_epochs=num_epochs, dataloaders_dict=dataloaders_dict, path_checkpoints=path_checkpoints, mode=mode, board_folder=experimentConfig, split_type=split_type) 
    
    
  
    

# __main__()
__main_mask__()






