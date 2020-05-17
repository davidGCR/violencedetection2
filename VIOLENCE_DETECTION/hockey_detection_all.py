
import sys
import include
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


from violenceDataset import *
# import trainer
# from tester import Tester
# from kfolds import *
from operator import itemgetter
import random
# from initializeModel import *
# from parameters import *
from transforms import *
import initializeDataset
import constants
import pandas as pd
# from FPS import FPSMeter
from torch.utils.tensorboard import SummaryWriter
import UTIL

def cv_it_accuracy(predictions, gt_labels):
    running_corrects = np.sum(predictions == gt_labels)
    acc = running_corrects / gt_labels.shape[0]
    return acc

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
    
import SALIENCY.saliencyModel
from SALIENCY.loss import Loss
from tqdm import tqdm
from torch.autograd import Variable
import include
import LOCALIZATION.localization_utils as localization_utils
from YOLOv3 import yolo_inference
# import LOCALIZATION.MaskRCNN as MaskRCNN
# import LOCALIZATION.point 

def train_mask_model(num_epochs, regularizers, device, checkpoint_path, dataloaders_dict, black_box_file, numDynamicImages):
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
            mask, out = saliency_m(inputs, labels)
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
    parser.add_argument("--numDynamicImagesPerVideo", type=int, help="num dyn imgs")
    parser.add_argument("--videoSegmentLength", type=int, default=0)
    parser.add_argument("--positionSegment", type=str)
    parser.add_argument("--shuffle", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--frame_skip", type=int)
    parser.add_argument("--overlaping", type=float)
    parser.add_argument("--split_type", type=str)

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
    transforms = createTransforms(input_size)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
                                            transforms, args.numDynamicImagesPerVideo, train_batch_size=args.batchSize, test_batch_size=1,
                                            train_num_workers=args.numWorkers, test_num_workers=1, videoSegmentLength=args.videoSegmentLength,
                                            positionSegment=args.positionSegment, overlaping=args.overlaping, frame_skip=args.frame_skip)
    
    experimentConfig = 'Mask_model, segmentLen-{}, numDynIms-{}, frameSkip-{}, epochs-{}'.format(args.videoSegmentLength,
                                                                                                                args.numDynamicImagesPerVideo,
                                                                                                                args.frame_skip,
                                                                                                                args.numEpochs,
                                                                                                                )
    # experimentConfig=''
    path_checkpoints = os.path.join(include.root, constants.HOCKEY_PATH_CHECKPOINTS, experimentConfig)
    black_box_file = os.path.join(include.root,constants.HOCKEY_PATH_CHECKPOINTS, 'HOCKEY-Model-resnet50, segmentLen-10, numDynIms-1, frameSkip-1, epochs-25, split_type-train-test.tar')

    train_mask_model(args.numEpochs, regularizers, device, path_checkpoints, dataloaders_dict, black_box_file, args.numDynamicImagesPerVideo)

def getPersonDetectorModel(detector_type, device):
    classes = None
    if detector_type == constants.YOLO:
        img_size = 416
        weights_path = include.root+"/YOLOv3/weights/yolov3.weights"
        class_path = include.root+"/YOLOv3/data/coco.names"
        model_def = include.root+"/YOLOv3/config/yolov3.cfg"
        person_model, classes = yolo_inference.initializeYoloV3(img_size, class_path, model_def, weights_path, device)
        
        # print('persons_in_frame MaskRCNN: ', len(persons_in_frame))
    # elif detector_type == constants.MASKRCNN:
    #     person_model = maskRCNN()
        

    return person_model, classes

def localization(persons_in_frame, thresh_close_persons, saliency_bboxes, iou_threshold = 0.3):
    persons_filtered, only_joined_regions = localization_utils.filterClosePersonsInFrame(persons_in_frame, thresh_close_persons)
    persons_filtered.sort(key=lambda x: x.iou, reverse=True)
    temporal_ious_regions = []
    anomalous_regions = []
    # for personBox in persons_filtered:
    personBox = persons_filtered[0]
    for saliencyBox in saliency_bboxes:
        iou = localization_utils.IOU(personBox, saliencyBox)
        # tmp_rgion = localization_utils.joinBBoxes(saliencyBox,personBox) #Nooo si el saliente bbox es todo frame
        tmp_rgion = personBox
        tmp_rgion.score = iou
        temporal_ious_regions.append(tmp_rgion)
        # print('----IOU (person and dynamic region): ', str(iou))
        if iou >= iou_threshold:
            abnormal_region = localization_utils.joinBBoxes(saliencyBox,personBox)
            abnormal_region.score = iou
            anomalous_regions.append(abnormal_region)
        # else:
        #     inside = localization_utils.intersetionArea(personBox, saliencyBox)
        #     if inside == personBox.area:
        #         abnormal_region = saliencyBox
        #         abnormal_region.score = iou
        #         anomalous_regions.append(abnormal_region)
        
    if len(anomalous_regions) > 1:
        anomalous_regions.sort(key=lambda x: x.iou, reverse=True) #sort by iuo to get only one anomalous regions
        anomalous_regions[0].score = -1
        # bbox_last = anomalous_regions[0]
    #     break
    # else: # 
    #     temporal_ious_regions.sort(key=lambda x: x.iou, reverse=True)
    #     anomalous_regions.append(temporal_ious_regions[0])
    #     break
    
    return persons_filtered, anomalous_regions


def __main_spatio_temporal__():
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

    
    videoSegmentLength = 10
    frame_skip = 0
    overlaping = 0
    positionSegment = 'begin'
    input_size = 224
    numDiPerVideos = 1
    transforms = createTransforms(input_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    h = 288
    w = 360
    type_person_detector = 'yolov3'
    person_model, classes = getPersonDetectorModel(type_person_detector, device)

    dataset = ViolenceDataset(dataset=test_x, labels=test_y, numFrames=test_numFrames , spatial_transform=transforms["val"], numDynamicImagesPerVideo=numDiPerVideos,
        videoSegmentLength=videoSegmentLength, positionSegment=positionSegment, overlaping=overlaping, frame_skip=frame_skip)
    
    video = '/Users/davidchoqueluqueroman/Desktop/PAPERS-CODIGOS/violencedetection2/HockeyFightsDATASET/frames/violence/25'
    videos_violence = [video]
    classifier_file = os.path.join(constants.HOCKEY_PATH_CHECKPOINTS,'HOCKEY-Model-resnet50, segmentLen-10, numDynIms-1, frameSkip-1, epochs-25, split_type-train-test.tar')
    mask_file = os.path.join(constants.HOCKEY_PATH_CHECKPOINTS,'Mask_model, segmentLen-10, numDynIms-1, frameSkip-1, epochs-10')

    classifier_model = load_model_inference(classifier_file,device)
    mask_model = load_model_inference(mask_file, device)

    # videos_violence = []
    # for idx,video in enumerate(test_x):
    #     if test_y[idx] == 1:
    #         videos_violence.append(video)

    for video in videos_violence:
        dynamicImages, label, idx_next_block, preprocessing_time, frames_list = dataset.getTemporalBlock(video, idx_next_block=0) # torch.Size([1, 3, 224, 224]), label: 1 
        frames2gif =[]
        while idx_next_block <= 40:
            print(frames_list)
            ########### PREDICTION
            dynamic_img = dynamicImages.to(device)
            with torch.set_grad_enabled(False):
                outputs = classifier_model(dynamic_img)
                p = torch.nn.functional.softmax(outputs, dim=1)
                p = p.cpu().numpy()
                values, indices = torch.max(outputs, 1)
                prediction = indices.cpu().item()
                score = p[0][1]
            print('Prediction: ', prediction)

            ########### MASK
            label = torch.from_numpy(np.array(label))
            label = label.to(device)
            di_images = dynamic_img.to(device)
            di_images, labels = Variable(di_images), Variable(label)
            masks, _ = mask_model(di_images, label)

            mascara = masks.detach().cpu()#(1, 1, 224, 224)
            mascara = torch.squeeze(mascara,0)#(1, 224, 224)
            mascara = min_max_normalize_tensor(mascara)  #to 0-1
            mascara = mascara.numpy().transpose(1, 2, 0)

            mascara = cv2.resize(mascara, dsize=(w, h), interpolation=cv2.INTER_CUBIC)

            ############ LOCALIZATION
            saliency_bboxes, preprocesing_outs, contours, hierarchy = localization_utils.computeBoundingBoxFromMask(mascara)  #only one by segment
            real_frames = []
            for i, frame_gt in enumerate(frames_list):
                frame_path = os.path.join(video,frame_gt)
                frame = Image.open(frame_path)
                frame = np.array(frame)
                real_frames.append(frame)
            
            frame = real_frames[int(len(real_frames) / 2)]
            persons_in_frame = []
            persons_in_segment = []
            if type_person_detector == constants.YOLO:
                img_size = 416
                conf_thres = 0.8
                nms_thres = 0.4
                persons_in_frame, person_detection_time = localization_utils.personDetectionInFrameYolo(person_model, img_size, conf_thres,
                                                                                nms_thres, classes, frame, device)
            # elif type_person_detector == constants.MASKRCNN:
            #     mask_rcnn_threshold = 0.4
            #     persons_in_frame, person_detection_time = MaskRCNN.personDetectionInFrameMaskRCNN(person_model, frame, mask_rcnn_threshold)
            anomalous_regions = []
            if len(persons_in_frame) > 0:
                persons_in_segment.append(persons_in_frame)
                thres_intersec_person_saliency = 0.5
                thresh_close_persons = 0.5
                persons_filtered, anomalous_regions = localization(persons_in_frame, thresh_close_persons, saliency_bboxes, thres_intersec_person_saliency)
                persons_in_segment.append(persons_filtered)
                print('Persons in frame {}, Persons filtered {}'.format(len(persons_in_frame), len(persons_filtered)))
            ########### PLOT 
            red = (0, 0, 255)
            green = (0, 255, 0)
            blue = (255, 0, 0)
            yellow = (0, 255, 255)
            magenta = (255,50,255)
            

            dynamic_img = min_max_normalize_tensor(dynamic_img) 
            dynamic_img = dynamic_img.numpy()[0].transpose(1, 2, 0)

            BORDER = np.ones((h, 20, 3))
            preprocesing_outs[0] = np.stack((preprocesing_outs[0],)*3, axis=-1)
            preprocesing_outs[1] = np.stack((preprocesing_outs[1],)*3, axis=-1)
            preprocess = np.concatenate((BORDER,
                                        preprocesing_outs[0], BORDER,
                                        preprocesing_outs[1], BORDER,
                                        preprocesing_outs[2], BORDER,
                                        preprocesing_outs[3],BORDER), axis=1)

            
            for idx, image in enumerate(real_frames):
                image_saliencies = image.copy()
                for s_bbox in saliency_bboxes:
                    cv2.rectangle(image_saliencies, (int(s_bbox.pmin.x), int(s_bbox.pmin.y)), (int(s_bbox.pmax.x), int(s_bbox.pmax.y)), yellow, 2)
                # print('image: ', type(image))
                if len(persons_in_frame)>0:
                    for person in persons_in_frame:
                        cv2.rectangle(image_saliencies, (int(person.pmin.x), int(person.pmin.y)), (int(person.pmax.x), int(person.pmax.y)), green, 2)
                if len(anomalous_regions) > 0:
                    pred = anomalous_regions[0]
                    cv2.rectangle(image_saliencies, (int(pred.pmin.x), int(pred.pmin.y)), (int(pred.pmax.x), int(pred.pmax.y)), blue, 3)
                
                img = cv2.cvtColor(image_saliencies, cv2.COLOR_BGR2RGB)
                im_pil = Image.fromarray(img)
                frames2gif.append(im_pil)

                cv2.imshow('image', image)    
                cv2.imshow('Dynamic Image', dynamic_img)
                cv2.imshow('Mask Image', mascara)
                cv2.imshow('Saliency regions', image_saliencies)
                cv2.imshow('preprocess', preprocess)

                pos_x = 20
                sep = 400
                cv2.namedWindow("image");#x,y
                cv2.moveWindow("image", pos_x, 100);

                cv2.namedWindow("Dynamic Image");
                cv2.moveWindow("Dynamic Image", pos_x+sep, 100);
                
                cv2.namedWindow("Mask Image");
                cv2.moveWindow("Mask Image", pos_x + 2 * sep, 100);
                
                cv2.namedWindow("Saliency regions");
                cv2.moveWindow("Saliency regions", pos_x + 3 * sep, 100);
                
                cv2.namedWindow("preprocess");
                cv2.moveWindow("preprocess", 20,500);
                if cv2.waitKey(50) & 0xFF == ord('q'):
                    break
            _, gif_name = os.path.split(video)
            
            # print('Block dynamic image size: {}, label: {} '.format(dynamicImages.size(),label))
            dynamicImages, label, idx_next_block, preprocessing_time, frames_list = dataset.getTemporalBlock(video, idx_next_block=idx_next_block)
        
        frames2gif[0].save(os.path.join(constants.HOCKEY_PATH_GIFTS,gif_name+'.gif'), save_all=True, append_images=frames2gif[1:],optimize=False, duration=40, loop=0)




# __main_spatio_temporal__()

