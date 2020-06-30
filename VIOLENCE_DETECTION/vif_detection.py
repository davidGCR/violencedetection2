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
from tqdm import tqdm
import scipy.io as sio

from UTIL.util import video2Images2, sortListByStrNumbers, save_csvfile_multicolumn
from UTIL.chooseModel import initialize_model, initialize_FCNN
from UTIL.parameters import verifiParametersToTrain
from VIOLENCE_DETECTION.violenceDataset import ViolenceDataset
from VIOLENCE_DETECTION.transforms import vifTransforms, compute_mean_std

from VIOLENCE_DETECTION.datasetsMemoryLoader import getFoldData, train_test_iteration, vifLoadData
from UTIL.trainer import Trainer
from dataloader import MyDataloader
from UTIL.resultsPolicy import ResultPolicy
from constants import DEVICE

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
    parser.add_argument("--overlapping", type=float)
    parser.add_argument("--frameSkip", type=int, default=0)
    parser.add_argument("--saveCheckpoint", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--segmentPreprocessing", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--split_type", type=str)
    parser.add_argument("--transferModel", type=str, default=None)
    args = parser.parse_args()

    # my_dict = args.__dict__
    # print(my_dict)
    
    cv_test_accs = []
    cv_test_losses = []
    cv_final_epochs = []
    

    
    if args.split_type == 'fully-conv':
        datasetAll, labelsAll, numFramesAll, splitsLen = vifLoadData(constants.PATH_VIF_FRAMES)
        transforms = vifTransforms(input_size=224)
        print('splitsLen=', splitsLen)
        default_args = {
                'X': datasetAll,
                'y': labelsAll,
                'numFrames': numFramesAll,
                'transform': transforms['val'],
                'NDI': args.numDynamicImagesPerVideo,
                'videoSegmentLength': args.videoSegmentLength,
                'positionSegment': args.positionSegment,
                'overlapping': args.overlapping,
                'frameSkip': args.frameSkip,
                'skipInitialFrames': 0,
                'batchSize': args.batchSize,
                'shuffle': False,
                'numWorkers': args.numWorkers,
                'pptype': None,
                'modelType': args.modelType
        }
        dt_loader = MyDataloader(default_args)
        model, _ = initialize_model(model_name=args.modelType,
                                    num_classes=2,
                                    feature_extract=args.featureExtract,
                                    numDiPerVideos=args.numDynamicImagesPerVideo,
                                    joinType=args.joinType,
                                    use_pretrained=True)
        tf_model = False
        if args.transferModel is not None:
            tf_model = True
            if DEVICE == 'cuda:0':
                model.load_state_dict(torch.load(args.transferModel), strict=False)
            else:
                model.load_state_dict(torch.load(args.transferModel, map_location=DEVICE))
        
        model = initialize_FCNN(model_name=args.modelType, original_model=model)
        # print(model)
        model.eval()
        outs = []
        labels = []
        for data in tqdm(dt_loader.dataloader):
            inputs, y, _, _ = data
            inputs = inputs.to(DEVICE)
            # print(inputs.size())
            y = y.to(DEVICE)
            with torch.no_grad():
                outputs = model(inputs)
                # print(outputs.size())
                outs.append(outputs)
                labels.append(y)
        print('outs_loader=shape {}, type {}'.format(len(outs), type(outs)))
        outs = torch.stack(outs, dim=0)
        print('outs=', outs.size(), type(outs))
        it, bacth, C, H, W = outs.size()
        outs = outs.view(it * bacth, C, H, W)
        outs = outs.permute(0, 2, 3, 1)
        
        n, H, W, C = outs.size()
        outs = outs.contiguous()
        outs = outs.view(n * H * W, C)
        # print('outs=', outs.size(), type(outs))
        outs = outs.numpy()
        # print('labels list=',len(labels))
        labels = torch.cat(labels, dim=0)
        labels = labels.numpy()
        # print('labels=', labels.shape, type(labels))
        print('outs=', outs.shape, type(outs))
        # print(labels)
        # print('conv5_train_test({})=shape {}, type {}'.format(i+1,outs.shape, type(outs)))
        name = os.path.join('/Users/davidchoqueluqueroman/Google Drive/ITQData','vif-{}-ndi={}-len={}-tfModel={}.mat'.format(args.modelType,args.numDynamicImagesPerVideo,args.videoSegmentLength, tf_model))
        sio.savemat(file_name=name,mdict={'fmaps':outs, 'labels':labels})
    elif args.split_type == 'cross-val':
        print(args.split_type)
        for fold in range(5):
            fold_path = os.path.join(constants.PATH_VIF_FRAMES,str(fold+1))
            train_x, train_y, train_numFrames, test_x, test_y, test_numFrames = train_test_iteration(fold_path, shuffle=True)
            # train = zip(train_names, train_labels, train_num_frames)
            # test = zip(test_names, test_labels, test_num_frames)
            # txt_file = 'Iteration({}).txt'.format(fold + 1)
            # save_csvfile_multicolumn(train, 'train-' + txt_file)
            # save_csvfile_multicolumn(test, 'test-'+txt_file)
            default_args = {
                'X': train_x,
                'y': train_y,
                'numFrames': train_numFrames,
                'transform': None,
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
            test_dt_loader = MyDataloader(default_args)
            # pop_mean, pop_std0, pop_std1 = compute_mean_std(train_dt_loader.dataloader)
            # print('Train-mean={}, std0={}, std1={}'.format(pop_mean, pop_std0, pop_std1))
            # pop_mean_test, pop_std0_test, pop_std1_test = compute_mean_std(test_dt_loader.dataloader)
            # print('Test-mean={}, std0={}, std1={}'.format(pop_mean_test, pop_std0_test, pop_std1_test))
            transforms = vifTransforms(input_size=224, train_mean=[0.5168978,  0.51586777, 0.5158742], train_std=[0.12358205, 0.11996705, 0.11759791], test_mean=[0.5168978,  0.51586777, 0.5158742], test_std=[0.12358205, 0.11996705, 0.11759791])
            train_dt_loader.transform = transforms['train']
            # print('Dataloader',train_dt_loader.dataloader)
            test_dt_loader.transform = transforms['val']
            # print('Dataloader',train_dt_loader.dataloader)
            model, _ = initialize_model(model_name=args.modelType,
                                        num_classes=2,
                                        feature_extract=args.featureExtract,
                                        numDiPerVideos=args.numDynamicImagesPerVideo,
                                        joinType=args.joinType,
                                        use_pretrained=True)
            model.to(DEVICE)
            if args.transferModel is not None:
                print('Transfering model ...')
                if DEVICE == 'cuda:0':
                    model.load_state_dict(torch.load(args.transferModel), strict=False)
                else:
                    model.load_state_dict(torch.load(args.transferModel, map_location=DEVICE))
            
            params_to_update = verifiParametersToTrain(model, args.featureExtract)
            optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
            criterion = nn.CrossEntropyLoss()
            experimentConfig = 'VIF-Model-{},segmentLen-{},numDynIms-{},frameSkip-{},epochs-{},splitType-{},fold-{}'.format(args.modelType,
                                                                                                                            args.videoSegmentLength,
                                                                                                                            args.numDynamicImagesPerVideo,
                                                                                                                            args.frameSkip,
                                                                                                                            args.numEpochs,
                                                                                                                            args.splitType,
                                                                                                                            str(fold+1))        
            log_dir = os.path.join(constants.PATH_RESULTS, 'VIF', 'tensorboard-runs', experimentConfig)
            writer = SummaryWriter(log_dir)
            tr = Trainer(model=model,
                        model_transfer= None,
                        train_dataloader=train_dt_loader.dataloader,
                        val_dataloader=test_dt_loader.dataloader,
                        criterion=criterion,
                        optimizer=optimizer,
                        num_epochs=args.numEpochs,
                        checkpoint_path=os.path.join(constants.PATH_RESULTS, 'VIF', 'checkpoints', experimentConfig+'.tar'),
                        lr_scheduler=None)
            
            policy = ResultPolicy()
            for epoch in range(1, args.numEpochs + 1):
                print("Fold {} ----- Epoch {}/{}".format(fold+1,epoch, args.numEpochs))
                # Train and evaluate
                epoch_loss_train, epoch_acc_train = tr.train_epoch(epoch)
                epoch_loss_val, epoch_acc_val = tr.val_epoch(epoch)
                exp_lr_scheduler.step()
                flac = policy.update(epoch_loss_train, epoch_acc_train, epoch_loss_val, epoch_acc_val, epoch)
                if args.saveCheckpoint:
                    tr.saveCheckpoint(epoch, flac, epoch_acc_val, epoch_loss_val)
                writer.add_scalar('training loss', epoch_loss_train, epoch)
                writer.add_scalar('validation loss', epoch_loss_val, epoch)
                writer.add_scalar('training Acc', epoch_acc_train, epoch)
                writer.add_scalar('validation Acc', epoch_acc_val, epoch)
            cv_test_accs.append(policy.getFinalTestAcc())
            cv_test_losses.append(policy.getFinalTestLoss())
            cv_final_epochs.append(policy.getFinalEpoch())
        print('CV Accuracies=', cv_test_accs)
        print('CV Losses=', cv_test_losses)
        print('CV Epochs=', cv_final_epochs)
        print('Test AVG Accuracy={}, Test AVG Loss={}'.format(np.average(cv_test_accs), np.average(cv_test_losses)))
        print("Accuracy: %0.2f (+/- %0.2f), Losses: %0.2f" % (np.array(cv_test_accs).mean(), np.array(cv_test_accs).std() * 2, np.array(cv_test_losses).mean()))  

__main__()