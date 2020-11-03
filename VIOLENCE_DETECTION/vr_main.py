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
from VIOLENCE_DETECTION.UTIL2 import base_dataset, load_model, transforms_dataset
from collections import Counter
from sklearn.model_selection import train_test_split

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
                inputs, dynamicImages, labels, v_names, one_box, paths = data
                # pts, dynamicImages, label, vid_name, gt_bboxes, torch.from_numpy(np.array(one_box)).float(), paths
                # print(v_names)
                # print('inputs=', inputs.size(), type(inputs))
                # print('one_box=', one_box.size(), type(one_box))

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
                    outputs = model(inputs, one_box)
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

def __weakly_localization__():
    from VIOLENCE_DETECTION.CAM import compute_CAM, cam2bb
    from VIOLENCE_DETECTION.datasetsMemoryLoader import load_fold_data
    from VIDEO_REPRESENTATION.dynamicImage import getDynamicImage
    from VIOLENCE_DETECTION.metrics import loc_error
    from MODELS.ViolenceModels import ResNet_ROI_Pool, ResNet
    import cv2

    ### Load model
    # modelPath = os.path.join(
    #     constants.PATH_RESULTS,
    #     'HOCKEY/checkpoints',
    #     'DYN_Stream-_dataset=[hockey]_model=resnet50_numEpochs=25_freezeConvLayers=True_numDynamicImages=1_segmentLength=30_frameSkip=0_skipInitialFrames=0_overlap=0.0_joinType=maxTempPool_useKeyframes=None_windowLen=0-fold=1.pt'
    #     )
    # dataset='hockey'
    # fold = 1

    # modelPath = os.path.join(
    #     constants.PATH_RESULTS,
    #     'UCFCRIME2LOCAL/checkpoints',
    #     'DYN_Stream-_dataset=[ucfcrime2local]_model=resnet50_numEpochs=25_freezeConvLayers=True_numDynamicImages=1_segmentLength=30_frameSkip=0_skipInitialFrames=0_overlap=0.0_joinType=maxTempPool_useKeyframes=None_windowLen=0-fold=4.pt'
    #     )

    # modelPath_rgb = os.path.join(
    #     constants.PATH_RESULTS,
    #     'UCFCRIME2LOCAL/checkpoints',
    #     'RGBCNN-dataset=ucfcrime2local_model=resnet50_frameIdx=14_numEpochs=25_featureExtract=False_fold=4.pt'
    #     )
    dataset='ucfcrime2local'
    fold = 4

    # stream_type='dyn_img'
    # modelPath = os.path.join(
    #     constants.PATH_RESULTS,
    #     'RWF-2000/checkpoints',
    #     'DYN_Stream-_dataset=[rwf-2000]_model=resnet50_numEpochs=25_freezeConvLayers=True_numDynamicImages=1_segmentLength=30_frameSkip=0_skipInitialFrames=0_overlap=0.0_joinType=maxTempPool_useKeyframes=None_windowLen=0-fold=1.pt'
    #     )
    # dataset='rwf-2000'
    # fold = 1

    # modelPath = os.path.join(
    #     constants.PATH_RESULTS,
    #     'VIF/checkpoints',
    #     'DYN_Stream-_dataset=[vif]_model=resnet50_numEpochs=25_freezeConvLayers=True_numDynamicImages=1_segmentLength=30_frameSkip=0_skipInitialFrames=0_overlap=0.0_joinType=maxTempPool_useKeyframes=None_windowLen=0-fold=1.pt'
    #     )
    # dataset='vif'
    # fold = 1

    if dataset == 'rwf-2000':
        train_x, train_y, train_numFrames, test_x, test_y, test_numFrames, transforms = base_dataset(dataset)
    else:
        datasetAll, labelsAll, numFramesAll, transforms = base_dataset(dataset)
        train_idx, test_idx = load_fold_data(dataset, fold=fold)
        # train_x = list(itemgetter(*train_idx)(datasetAll))
        # train_y = list(itemgetter(*train_idx)(labelsAll))
        # train_numFrames = list(itemgetter(*train_idx)(numFramesAll))
        test_x = list(itemgetter(*test_idx)(datasetAll))
        test_y = list(itemgetter(*test_idx)(labelsAll))
        test_numFrames = list(itemgetter(*test_idx)(numFramesAll))

    # model_, args = load_model(modelPath, stream_type='dyn_img')
    # test_dataset = ViolenceDataset(videos=test_x,
    #                                 labels=test_y,
    #                                 numFrames=test_numFrames,
    #                                 spatial_transform=transforms['val'],
    #                                 positionSegment=None,
    #                                 overlaping=args['overlapping'],
    #                                 frame_skip=args['frameSkip'],
    #                                 skipInitialFrames=args['skipInitialFrames'],
    #                                 ppType=None,
    #                                 useKeyframes=args['useKeyframes'],
    #                                 windowLen=args['windowLen'],
    #                                 numDynamicImagesPerVideo=args['numDynamicImages'], #
    #                                 videoSegmentLength=args['videoSegmentLength'],
    #                                 dataset=dataset
    #                                 )
 
    model_2 = ResNet_ROI_Pool(

        )
    initialize_model(model_name='resnet50',
                                        num_classes=2,
                                        freezeConvLayers=True,
                                        numDiPerVideos=1,
                                        joinType='maxTempPool',
                                        use_pretrained=True)
    test_dataset = ViolenceDataset(videos=test_x,
                                    labels=test_y,
                                    numFrames=test_numFrames,
                                    spatial_transform=transforms['val'],
                                    numDynamicImagesPerVideo=1,
                                    videoSegmentLength=30,
                                    dataset=dataset
                                    )

   
    # print(model_2)
    # test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=True, num_workers=4)
    # indices =  [8, 0, 1, 2, 9, 11, 3, 5]  # select your indices here as a list  
    # subset = torch.utils.data.Subset(test_dataset, indices)
    
    def background_model(dynamicImages, iters=10):
        # img_0 = dynamicImages[0]
        # img_0 = torch.squeeze(img_0).numpy()
        masks = []
        backSub = cv2.createBackgroundSubtractorKNN(detectShadows=True)
        for i in range(iters):
            if i<len(dynamicImages):
                img_1 = dynamicImages[i]
            else:
                img_1 = dynamicImages[len(dynamicImages)-1]
            img_1 = torch.squeeze(img_1).numpy()
            img_1 = cv2.fastNlMeansDenoisingColored(img_1,None,10,10,7,21)

            fgMask = backSub.apply(img_1)
            print('---iter ({})'.format(i+1))

            cv2.imshow("img_1", img_1)
            key = cv2.waitKey(0)

            # cv2.imshow("fgMask", fgMask)
            # key = cv2.waitKey(0)

            # gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
            # cv2.imshow("gray", gray)
            # key = cv2.waitKey(0)

            # threshold = 0.60*np.amax(gray)
            # ret, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            
            
            # threshold_inv = np.amin(gray) + 60
            # ret, thresh_inv = cv2.threshold(gray, threshold_inv, 255, cv2.THRESH_BINARY)

            # print('max={}, min={}, mean={}, threshold={}, thresh_inv={}'.format(np.amax(gray), np.amin(gray), np.mean(gray), threshold, threshold_inv))
            
            # cv2.imshow("thresh", thresh)
            # key = cv2.waitKey(0)

            # cv2.imshow("thresh_inv", thresh_inv)
            # key = cv2.waitKey(0)

            

            # diff = img_1-img_0
            # cv2.imshow("diff", diff)
            # key = cv2.waitKey(0)

            # img_0 = img_1
        # kernel = np.ones((5, 5), np.uint8)
        # fgMask_d = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)
        fgMask_e = cv2.erode(fgMask,np.ones((3, 3), np.uint8),iterations = 1)
        fgMask_d = cv2.dilate(fgMask_e,np.ones((7, 7), np.uint8),iterations = 1)
        cv2.imshow("fgMask", fgMask)
        cv2.imshow("fgMask_d", fgMask_d)
        key = cv2.waitKey(0)

    ## Analysis
    # test_dataset.numDynamicImagesPerVideo = 20
    # test_dataset.videoSegmentLength = 5
    # test_dataset.overlapping = 0.5
    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = 1, shuffle =True, num_workers=1)

    # print(test_dataset.numDynamicImagesPerVideo, test_dataset.videoSegmentLength)
    # for data in dataloader:
    #     inputs, dynamicImages, labels, v_names, _, paths = data
    #     # print('len(dynamicImages)=',len(dynamicImages))
    #     print(v_names)
    #     print(inputs.size())
    #     background_model(dynamicImages, iters=20)
        # print(len(dynamicImages), type(dynamicImages[0]), dynamicImages[0].size())
        # for i, dyn_img in enumerate(dynamicImages):
        #     dyn_img = torch.squeeze(dyn_img).numpy()
        #     # print('Image-{}='.format(i+1), dyn_img.shape)
        #     cv2.imshow("dyn_img", dyn_img)
        #     key = cv2.waitKey(0)

        #     dst = cv2.fastNlMeansDenoisingColored(dyn_img,None,10,10,7,21)
        #     cv2.imshow("dyn_image denoised", dst)
        #     key = cv2.waitKey(0)

        #     gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        #     cv2.imshow("Gray", gray)
        #     key = cv2.waitKey(0)

        #     heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        #     cv2.imshow("heatmap", heatmap)
        #     key = cv2.waitKey(0)

        #     threshold = 0.60*np.amax(gray)
        #     ret, thresh1 = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        #     cv2.imshow("threshold", thresh1)
        #     key = cv2.waitKey(0)
            

    # ## Dynamic Image CNN
    y, y_preds = [], []
    for data in dataloader:
        # ipts, dynamicImages, label, vid_name
        inputs, dynamicImages, labels, v_names, gt_bboxes, one_box, paths = data
        print(v_names)
        print('---inputs=', inputs.size(), inputs.type())
        print('---one_box=', one_box.size(), one_box.type())
        print('---gt_bboxes=', len(gt_bboxes), type(gt_bboxes))

        # for g in gt_bboxes:
        #     print(g)
        
        # y = model_2(inputs, one_box)
        # print('y=',type(y))

    #     dyn_img = torch.squeeze(dynamicImages[0]).numpy()
    #     print('Image1=', dyn_img.shape)

    #     cv2.imshow("dyn_img_1", dyn_img)
    #     key = cv2.waitKey(0)

    #     gray = cv2.cvtColor(dyn_img, cv2.COLOR_BGR2GRAY)
    #     cv2.imshow("Gray", gray)
    #     key = cv2.waitKey(0)

    #     heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    #     cv2.imshow("Image", heatmap)
    #     key = cv2.waitKey(0)

    #     print('v_names=',v_names)
    #     label = labels.item()
    #     if label == 1 and dataset=='ucfcrime2local':
    #         # gt_bboxes, one_box = load_bbox_gt(v_names[0], paths[0])
    #         y.append([label]+one_box)
    #     else:
    #         y.append([label]+[None, None, None, None])

        for i,pt in enumerate(paths[0]):
            print('----one_box=',one_box)
            frame = cv2.imread(pt[0])
            x0, y0, w, h = gt_bboxes[i]
            cv2.rectangle(frame, (x0, y0),(x0+w, y0+h), (0,255,0), 2)
            # x1=one_box[0,0]
            # y1=one_box[0,1]
            # x2=one_box[0,0]+one_box[0,2]
            # y2=one_box[0,1]+one_box[0,3]
            # cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
            cv2.imshow("frame", frame)
            key = cv2.waitKey(0)

    #     y_pred, CAM, heatmap = compute_CAM(model_, inputs, 'convLayers', dyn_img, plot=True)
    #     x0, y0, w, h = cam2bb(CAM, plot=False)
    #     y_preds.append([y_pred, x0, y0, w, h])

    # le = loc_error(y, y_preds)
    # print('Localization error={}'.format(le))

    # # RGB CNN
    # model_rgb, args_rgb = load_model(modelPath_rgb, stream_type='rgb')
    # rgb_transforms = transforms_dataset(dataset, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # rgb_dataset = RGBDataset(dataset=test_x,
    #                             labels=test_y,
    #                             numFrames=test_numFrames,
    #                             spatial_transform=rgb_transforms['test'],
    #                             frame_idx=args_rgb['frameIdx'])
    # rgb_dataloader = torch.utils.data.DataLoader(rgb_dataset, batch_size=1, shuffle=False, num_workers=4)

    
    # y, y_preds = [], []
    # for data in rgb_dataloader:
    #     v_name, inputs, labels, frame_path = data
    #     print('frame_path=',frame_path)
    #     label = labels.item()
    #     frame=cv2.imread(frame_path[0])
    #     if label == 1 and dataset=='ucfcrime2local':
    #         gt_bboxes, one_box = load_bbox_gt(v_name[0], [frame_path])
    #         y.append([label]+one_box)
    #     else:
    #         y.append([label]+[None, None, None, None])
    #         # net, x, final_conv, image, plot=False
    #     y_pred, CAM, heatmap = compute_CAM(model_rgb, inputs, 'layer4', frame, plot=False)
    #     x0, y0, w, h = cam2bb(CAM, plot=False)
    #     y_preds.append([y_pred, x0, y0, w, h])
    # le = loc_error(y, y_preds)
    # print('Localization error RGB={}'.format(le))

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
        if args.dataset[0] == 'rwf-2000':
            train_x, train_y, train_numFrames, test_x, test_y, test_numFrames, transforms = base_dataset(args.dataset[0])
            print(len(train_x), len(train_y), len(train_numFrames), len(test_x), len(test_y), len(test_numFrames))
        else:
            train_x, train_y, test_x, test_y = None, None, None, None
            train_x = list(itemgetter(*train_idx)(datasetAll))
            train_y = list(itemgetter(*train_idx)(labelsAll))
            train_numFrames = list(itemgetter(*train_idx)(numFramesAll))
            test_x = list(itemgetter(*test_idx)(datasetAll))
            test_y = list(itemgetter(*test_idx)(labelsAll))
            test_numFrames = list(itemgetter(*test_idx)(numFramesAll))

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
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=True, num_workers=args.numWorkers)

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
            checkpoint_path = args_2_checkpoint_path(args, fold=fold)
        
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

def skorch_a():
    from skorch import NeuralNetClassifier
    from skorch.helper import predefined_split

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
    parser.add_argument("--modelPath", type=str, default=None)
    parser.add_argument("--testDataset",type=str, default=None)
    parser.add_argument("--transferModel", type=str, default=None)

    args = parser.parse_args()
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
       
    for train_idx, test_idx in customize_kfold(n_splits=folds_number, dataset=args.dataset[0], X=datasetAll, y=labelsAll, shuffle=shuffle):
        fold = fold + 1
        print("**************** Fold:{}/{} ".format(fold, folds_number))
        if args.dataset[0] == 'rwf-2000':
            train_x, train_y, train_numFrames, test_x, test_y, test_numFrames, transforms = base_dataset(args.dataset[0])
            # print(len(train_x), len(train_y), len(train_numFrames), len(test_x), len(test_y), len(test_numFrames))
        else:
            train_x, train_y, test_x, test_y = None, None, None, None
            train_x = list(itemgetter(*train_idx)(datasetAll))
            train_y = list(itemgetter(*train_idx)(labelsAll))
            train_numFrames = list(itemgetter(*train_idx)(numFramesAll))
            test_x = list(itemgetter(*test_idx)(datasetAll))
            test_y = list(itemgetter(*test_idx)(labelsAll))
            test_numFrames = list(itemgetter(*test_idx)(numFramesAll))

        train_x, val_x, train_numFrames, val_numFrames, train_y, val_y = train_test_split(train_x, train_numFrames, train_y, test_size=0.2, stratify=train_y, random_state=1)
        #Label distribution
        print('Label distribution:')
        print('Train=', Counter(train_y))
        print('Val=', Counter(val_y))
        print('Test=', Counter(test_y))

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
        
        from MODELS.ViolenceModels import ResNet
        # PretrainedModel = ResNet(num_classes=2,
        #                         numDiPerVideos=args.numDynamicImagesPerVideo,
        #                         model_name=args.modelType,
        #                         joinType=args.joinType,
        #                         freezeConvLayers=args.freezeConvLayers) 
        
        PretrainedModel, _ = initialize_model(model_name=args.modelType,
                                        num_classes=2,
                                        freezeConvLayers=args.freezeConvLayers,
                                        numDiPerVideos=args.numDynamicImagesPerVideo,
                                        joinType=args.joinType,
                                        use_pretrained=True)

        from skorch.callbacks import LRScheduler
        lrscheduler = LRScheduler(policy='StepLR', step_size=7, gamma=0.1)
        
        from skorch.callbacks import Freezer
        freezer = Freezer(lambda x: not x.startswith('linear'))

        from skorch.callbacks import TensorBoard
        writer = SummaryWriter()
        ts = TensorBoard(writer)

        from skorch.callbacks import Checkpoint
        checkpoint_path = args_2_checkpoint_path(args, fold=fold)
        checkpoint = Checkpoint(f_params=checkpoint_path, monitor='valid_acc_best')

        net = NeuralNetClassifier(
                PretrainedModel, 
                criterion=nn.CrossEntropyLoss,
                lr=0.001,
                batch_size=args.batchSize,
                max_epochs=args.numEpochs,
                optimizer=optim.SGD,
                optimizer__momentum=0.9,
                iterator_train__shuffle=True,
                iterator_train__num_workers=4,
                iterator_valid__shuffle=True,
                iterator_valid__num_workers=4,
                train_split=predefined_split(val_dataset),
                callbacks=[lrscheduler, freezer, ts]
                # device=DEVICE
                # module__output_features=2,
            )
        net.fit(train_dataset, y=None);

        from sklearn.metrics import accuracy_score
        from skorch.helper import SliceDataset

        y_pred = net.predict(test_dataset)
        
        # test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.numWorkers)
        # y_test = [yy for _, yy in test_dataloader]
        
        acc = accuracy_score(SliceDataset(test_dataset, idx=1), y_pred)
        cv_test_accs.append(acc)
        print('Accuracy={}'.format(acc))
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
    # print('CV Losses=', cv_test_losses)
    # print('CV Epochs=', cv_final_epochs)
    print('Test AVG Accuracy={}'.format(np.average(cv_test_accs))
    print("Accuracy: %0.3f (+/- %0.3f)" % (np.array(cv_test_accs).mean(), np.array(cv_test_accs).std() * 2)

if __name__ == "__main__":
    skorch_a()
    # __my_main__()
    # __weakly_localization__()

    

