import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import include
import constants

import argparse
import numpy as np
import random
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from UTIL.util import video2Images2, sortListByStrNumbers, save_csvfile_multicolumn
from UTIL.initializeModel import initialize_model
from UTIL.parameters import verifiParametersToTrain
from VIOLENCE_DETECTION.violenceDataset import ViolenceDataset
import VIOLENCE_DETECTION.hockey_transforms as hockey_transforms
from UTIL.trainer import Trainer
# from UTIL.tester import Tester

def preprocessing_dataset(dataset_path_videos, dataset_path_frames):
    folds = os.listdir(dataset_path_videos)
    folds = sortListByStrNumbers(folds)
    for f in folds:
        violence_path = os.path.join(dataset_path_videos, f, 'Violence')
        non_violence_path = os.path.join(dataset_path_videos, f, 'NonViolence')
        violence_videos = os.listdir(violence_path)
        non_violence_videos = os.listdir(non_violence_path)
        # videos = violence_videos + non_violence_videos
        for video in violence_videos:
            video_folder = os.path.join(dataset_path_frames, str(f), 'Violence', video[:-4])
            video_full_path = os.path.join(violence_path,video)
            if not os.path.exists(video_folder):
                os.makedirs(video_folder)
            video2Images2(video_full_path, video_folder)
        
        for video in non_violence_videos:
            video_folder = os.path.join(dataset_path_frames, str(f), 'NonViolence', video[:-4])
            video_full_path = os.path.join(non_violence_path,video)
            if not os.path.exists(video_folder):
                os.makedirs(video_folder)
            video2Images2(video_full_path,video_folder)

def dataset_statistics(dataset_path_frames):
    folds = os.listdir(dataset_path_frames)
    folds = sortListByStrNumbers(folds)
    violence_x = []
    violence_y = []
    violence_num_frames = []
    nonviolence_num_frames = []

    for f in folds:
        violence_path = os.path.join(dataset_path_frames, f, 'Violence')
        non_violence_path = os.path.join(dataset_path_frames, f, 'NonViolence')
        violence_videos = os.listdir(violence_path)
        non_violence_videos = os.listdir(non_violence_path)
        for video in violence_videos:
            video_folder = os.path.join(violence_path, video)
            violence_num_frames.append(len(os.listdir(video_folder)))
            violence_x.append('fold:{}, video: {}, numFrames: {}, label: {}'.format(f, video_folder, len(os.listdir(video_folder)), 1))
        for video in non_violence_videos:
            video_folder = os.path.join(non_violence_path, video)
            nonviolence_num_frames.append(len(os.listdir(video_folder)))
            violence_x.append('fold:{}, video: {}, numFrames: {}, label: {}'.format(f, video_folder, len(os.listdir(video_folder)), 0))

    # Violence AVG: 101.869918699187, MAX: 161, MIN: 26
    # NonViolence AVG: 77.59349593495935, MAX: 119, MIN: 26
    # AVG_dataset:  89.73170731707317
    return violence_num_frames, nonviolence_num_frames, violence_x

def getFoldData(fold_path):
    names = []
    labels = []
    num_frames = []

    violence_path = os.path.join(fold_path, 'Violence')
    non_violence_path = os.path.join(fold_path, 'NonViolence')
    violence_videos = os.listdir(violence_path)
    non_violence_videos = os.listdir(non_violence_path)
    for video in violence_videos:
        video_folder = os.path.join(violence_path, video)
        num_frames.append(len(os.listdir(video_folder)))
        names.append(video_folder)
        labels.append(1)
    for video in non_violence_videos:
        video_folder = os.path.join(non_violence_path, video)
        num_frames.append(len(os.listdir(video_folder)))
        names.append(video_folder)
        labels.append(0)

    return names, labels, num_frames

def train_test_split(test_fold_path, shuffle):
    test_names, test_labels, test_num_frames = getFoldData(test_fold_path)
    folds_dir, fold_number = os.path.split(test_fold_path)
    fold_number = int(fold_number)
    train_names = []
    train_labels = []
    train_num_frames = []

    for i in range(5):
        if i + 1 != fold_number:
            names, labels, num_frames = getFoldData(os.path.join(folds_dir, str(i + 1)))
            train_names.extend(names)
            train_labels.extend(labels)
            train_num_frames.extend(num_frames)
    if shuffle:
        combined = list(zip(train_names, train_labels, train_num_frames))
        random.shuffle(combined)
        train_names[:], train_labels[:], train_num_frames[:] = zip(*combined)

        combined = list(zip(test_names, test_labels, test_num_frames))
        random.shuffle(combined)
        test_names[:], test_labels[:], test_num_frames[:] = zip(*combined)

    return train_names, train_labels, train_num_frames, test_names, test_labels, test_num_frames


def __main__():
    # violence_num_frames, nonviolence_num_frames, violence_x = dataset_statistics(constants.PATH_VIF_FRAMES)
    # avg_violence_frames = np.average(violence_num_frames) #101.869
    # avg_nonviolence_frames = np.average(nonviolence_num_frames) #77.59
    # avg_all = np.average(violence_num_frames+nonviolence_num_frames) #89.73
    # print('Violence AVG: {}, MAX: {}, MIN: {}'.format(avg_violence_frames, np.amax(violence_num_frames), np.amin(violence_num_frames)))
    # print('NonViolence AVG: {}, MAX: {}, MIN: {}'.format(avg_nonviolence_frames, np.amax(nonviolence_num_frames), np.amin(nonviolence_num_frames)))
    # print('AVG_dataset: ', avg_all)
    # print(violence_x)
    # preprocessing_dataset(constants.PATH_VIF_VIDEOS, constants.PATH_VIF_FRAMES)
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelType",type=str)
    parser.add_argument("--numEpochs",type=int)
    parser.add_argument("--batchSize",type=int)
    parser.add_argument("--featureExtract",type=lambda x: (str(x).lower() == 'true'), default=False, help="to fine tunning")
    parser.add_argument("--numDynamicImagesPerVideo", type=int)
    parser.add_argument("--joinType", type=str)
    parser.add_argument("--foldsNumber", type=int, default=5)
    parser.add_argument("--numWorkers", type=int, default=4)
    parser.add_argument("--videoSegmentLength", type=int)
    parser.add_argument("--positionSegment", type=str)
    parser.add_argument("--splitType", type=str)
    parser.add_argument("--overlaping", type=float)
    parser.add_argument("--frameSkip", type=int, default=0)
    args = parser.parse_args()

    data_transforms = hockey_transforms.createTransforms(224)

    for fold in range(5):
        fold_path = os.path.join(constants.PATH_VIF_FRAMES,str(fold+1))
        train_names, train_labels, train_num_frames, test_names, test_labels, test_num_frames = train_test_split(fold_path, shuffle=True)
        # train = zip(train_names, train_labels, train_num_frames)
        # test = zip(test_names, test_labels, test_num_frames)
        # txt_file = 'Iteration({}).txt'.format(fold + 1)
        # save_csvfile_multicolumn(train, 'train-' + txt_file)
        # save_csvfile_multicolumn(test, 'test-'+txt_file)
        image_datasets = {
            "train": ViolenceDataset(dataset=train_names,
                                    labels=train_labels,
                                    numFrames=train_num_frames,
                                    spatial_transform=data_transforms["train"],
                                    numDynamicImagesPerVideo=args.numDynamicImagesPerVideo,
                                    videoSegmentLength=args.videoSegmentLength,
                                    positionSegment=args.positionSegment,
                                    overlaping=args.overlaping,
                                    frame_skip=args.frameSkip),
            "val": ViolenceDataset(dataset=test_names,
                                    labels=test_labels,
                                    numFrames=test_num_frames,
                                    spatial_transform=data_transforms["val"],
                                    numDynamicImagesPerVideo=args.numDynamicImagesPerVideo,
                                    videoSegmentLength=args.videoSegmentLength,
                                    positionSegment=args.positionSegment,
                                    overlaping=args.overlaping,
                                    frame_skip=args.frameSkip)
        }
        dataloaders_dict = {
            "train": torch.utils.data.DataLoader(image_datasets["train"], batch_size=args.batchSize, shuffle=True, num_workers=args.numWorkers, pin_memory=True),
            "val": torch.utils.data.DataLoader(image_datasets["val"], batch_size=args.batchSize, shuffle=True, num_workers=args.numWorkers, pin_memory=True),
        }
        model, _ = initialize_model(model_name=args.modelType,
                                    num_classes=2,
                                    feature_extract=args.featureExtract,
                                    numDiPerVideos=args.numDynamicImagesPerVideo,
                                    joinType=args.joinType,
                                    use_pretrained=True)
        model.to(constants.DEVICE)
        params_to_update = verifiParametersToTrain(model, args.featureExtract)
        optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        criterion = nn.CrossEntropyLoss()
        experimentConfig = 'VIF-Model-{}, segmentLen-{}, numDynIms-{}, frameSkip-{}, epochs-{}, splitType-{}, fold-{}'.format(args.modelType,
                                                                                                                        args.videoSegmentLength,
                                                                                                                        args.numDynamicImagesPerVideo,
                                                                                                                        args.frameSkip,
                                                                                                                        args.numEpochs,
                                                                                                                        args.splitType,
                                                                                                                        str(fold+1))
        writer = SummaryWriter('RESULTS/Vif-tensorboard/'+experimentConfig)
        tr = Trainer(model=model,
                    train_dataloader=dataloaders_dict['train'],
                    val_dataloader=dataloaders_dict['val'],
                    criterion=criterion,
                    optimizer=optimizer,
                    scheduler=exp_lr_scheduler,
                    device=constants.DEVICE,
                    num_epochs=args.numEpochs,
                    checkpoint_path=experimentConfig,
                    numDynamicImage=args.numDynamicImagesPerVideo,
                    plot_samples=False,
                    train_type='train',
                    save_model=False)

        for epoch in range(1, args.numEpochs + 1):
            print("Fold {} ----- Epoch {}/{}".format(fold+1,epoch, args.numEpochs))
            # Train and evaluate
            epoch_loss_train, epoch_acc_train = tr.train_epoch(epoch)
            epoch_loss_val, epoch_acc_val = tr.val_epoch(epoch)
            exp_lr_scheduler.step()

            writer.add_scalar('training loss', epoch_loss_train, epoch)
            writer.add_scalar('validation loss', epoch_loss_val, epoch)
            writer.add_scalar('training Acc', epoch_acc_train, epoch)
            writer.add_scalar('validation Acc', epoch_acc_val, epoch)                                                                                                

__main__()