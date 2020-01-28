
import sys
# sys.path.insert(1, '/media/david/datos/PAPERS-SOURCE_CODE/violencedetection')
sys.path.insert(1,'/Users/davidchoqueluqueroman/Desktop/PAPERS-CODIGOS/violencedetection2')
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
from MaskDataset import MaskDataset
# from saliency_model import *
import initializeDataset
import constants
import pandas as pd
import LOCALIZATION.localization_utils as localization_utils
import SALIENCY.saliencyTester as saliencyTester

def online(violenceDataset, saliency_tester, type_person_detector, h, w, plot, only_video_name):
    person_model, classes = localization_utils.getPersonDetectorModel(type_person_detector)
    bbox_last = None
    distance_th = 35.5
    thres_intersec_lastbbox_current = 0.4
    data_rows = []
    indx_flac = -1
    num_video_segments = 0
    videos_scores = []

    for idx_video, data in enumerate(violenceDataset):
        indx_flac = idx_video
        if only_video_name is not None and indx_flac==0:
            if indx_flac==0:
                idx = anomalyDataset.getindex(only_video_name)
                if idx is not None:
                    data = anomalyDataset[idx]
                    idx_video = idx
                    print('TEsting only one video...' , only_video_name)
                else:
                    print('No valid video...')
            else:
                break

        if only_video_name is not None and indx_flac>0:
            break
        # else:
        #     break
        bbox_last = None
        print("-" * 150, 'video No: ', idx_video)
        #di_images = [1,ndis,3,224,224]
        video_name, label = data
        video_name = [video_name]
        label = torch.tensor(label)
        # data_rows_video = []
        #First Segment
        dis_images, segment_info, idx_next_segment = violenceDataset.computeSegmentDynamicImg(idx_video=idx_video, idx_next_segment=0)
        num_segment = 0
        while (dis_images is not None):
            num_segment += 1
            print(video_name, '---segment No: ', str(num_segment),'-----', dis_images.size(), len(segment_info))
            
    #         dis_images = torch.unsqueeze(dis_images, 0)
    #         segment_info = default_collate([segment_info])
    #         segment_info = np.array(segment_info)
    #         # dis_images = dis_images / 2 + 0.5 
    #         ttttt = saliency_tester.min_max_normalize_tensor(dis_images) 
    #         ttttt = ttttt.numpy()[0].transpose(1, 2, 0)
    #         ttttt = cv2.resize(ttttt, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
    #         # video_dynamic_images.append(ttttt)

    #         # plt.imshow(ttttt)
    #         # plt.title('Input for Classifier')
    #         # plt.show()
    #         # cv2.imshow('image', ttttt)
    #         # k = cv2.waitKey(0)
    #         # if k == 27:         # wait for ESC key to exit
    #         #     cv2.destroyAllWindows()
            

    #         ######################## mask
    #         masks = saliency_tester.compute_mask(dis_images, label)  #[1, 1, 224, 224]
    #         mascara = masks.detach().cpu()#(1, 1, 224, 224)
    #         mascara = torch.squeeze(mascara,0)#(1, 224, 224)
    #         mascara = saliency_tester.min_max_normalize_tensor(mascara)  #to 0-1
    #         mascara = mascara.numpy().transpose(1, 2, 0)
    #         ##resize 
    #         mascara = cv2.resize(mascara, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
    #         # video_mascaras.append(mascara)
            
    #         # plt.imshow(mascara_gray)
    #         # plt.title('MASCARA gray')
    #         # plt.show()

    #         # plt.imshow(img)
    #         # plt.title('MASCARA gray contours')
    #         # plt.show()
    #         # mascara = mascara_by_summarized[:,:,0]

            
    #         # mascara = ttttt[:,:,0]
    #         saliency_bboxes, preprocesing_outs, contours, hierarchy = localization_utils.computeBoundingBoxFromMask(mascara)  #only one by segment
    #         # video_preprocesing_outs.append(preprocesing_outs)
    #         # video_saliency_bboxes.append(saliency_bboxes)
            
    #         #metricas (no decir es mejor... 2 puntos porcentuales del valor-outperform) en esa base de datos- grafico explicar ejes...porque de los resultados
    #         #presentar errores
    #         saliency_bboxes = localization_utils.removeInsideSmallAreas(saliency_bboxes)            
    #         if bbox_last is not None:
    #             bboxes_with_high_core = []
    #             for bbox in saliency_bboxes:
    #                 pred_distance = localization_utils.distance(bbox.center, bbox_last.center)
    #                 if pred_distance <= distance_th or bbox.percentajeArea(localization_utils.intersetionArea(bbox,bbox_last)) >= thres_intersec_lastbbox_current:
    #                     bbox.score = pred_distance
    #                     bboxes_with_high_core.append(bbox)
    #             if len(bboxes_with_high_core) > 0:
    #                 bboxes_with_high_core.sort(key=lambda x: x.score)
    #                 saliency_bboxes = bboxes_with_high_core
    #                 # saliency_bboxes[0].score = -1
    #             else:
    #                 print('FFFFail in close lastBBox and saliency bboxes...')
    #                 saliency_bboxes = [bbox_last]

    #         if len(saliency_bboxes) == 0:
    #             saliency_bboxes = [BoundingBox(Point(0,0),Point(w,h))]
            
    #         frames_names, real_frames, real_bboxes = localization_utils.getFramesFromSegment(video_name[0], segment_info, 'all')
    #         # video_real_frames.append(real_frames)
    #         # video_real_bboxes.append(real_bboxes)
            
    #         persons_in_segment = []
    #         persons_segment_filtered = []
    #         anomalous_regions = []  # to plot
    #         segmentBox = localization_utils.getSegmentBBox(real_bboxes)
            
    #         print('Real frames shot: ', len(real_frames))
    #         temporal_ious_regions = []
    #         for idx, frame in enumerate(real_frames): #Detect persons frame by frame 
    #             print('Shot frame to process: ',idx)
    #             persons_in_frame = []
    #             persons_filtered = []
    #             if type_person_detector == constants.YOLO:
    #                 img_size = 416
    #                 conf_thres = 0.8
    #                 nms_thres = 0.4
    #                 persons_in_frame = localization_utils.personDetectionInFrameYolo(person_model, img_size, conf_thres, nms_thres, classes, frame)
    #             elif type_person_detector == constants.MASKRCNN:
    #                 mask_rcnn_threshold = 0.4
    #                 persons_in_frame = MaskRCNN.personDetectionInFrameMaskRCNN(person_model, frame, mask_rcnn_threshold)
    #             print('--num Persons in frame: ', len(persons_in_frame))
                
    #             if len(persons_in_frame) > 0:
    #                 persons_in_segment.append(persons_in_frame)
    #                 iou_threshold = 0.3
    #                 thres_intersec_person_saliency = 0.5
    #                 thresh_close_persons = 20
                
    #                 if bbox_last is not None:
    #                     print('='*10, ' ONLINE ')
    #                     dynamic_region = saliency_bboxes[0]
    #                     persons_filtered, only_joined_regions = localization_utils.filterClosePersonsInFrame(persons_in_frame, thresh_close_persons)
    #                     print('---Persons after filter close: ', len(persons_filtered), len(only_joined_regions))
    #                     # persons_segment_filtered.append(persons_filtered)
    #                     for person in persons_filtered:
    #                         iou = localization_utils.IOU(person, dynamic_region)
    #                         print('----IOU (person and dynamic region): ', str(iou))
    #                         if iou >= iou_threshold:
    #                             abnorm_bbox = localization_utils.joinBBoxes(person,dynamic_region)
    #                             abnorm_bbox.score = iou
    #                             anomalous_regions.append(abnorm_bbox)
                        
    #                     if len(anomalous_regions) > 0:
    #                         anomalous_regions.sort(key=lambda x: x.iou, reverse=True) #sort by iuo to get only one anomalous regions
    #                         anomalous_regions[0].score = -1
    #                         # bbox_last = anomalous_regions[0]
    #                         break

    #                 else:
    #                     print('='*10, ' FIRST SHOT ')
    #                     persons_filtered, only_joined_regions = localization_utils.filterClosePersonsInFrame(persons_in_frame, thresh_close_persons)
    #                     print('---Persons after filter close: ', len(persons_filtered), len(only_joined_regions))
    #                     # persons_segment_filtered.append(persons_filtered)
                        
    #                     for personBox in persons_filtered:
    #                         for saliencyBox in saliency_bboxes:
    #                             iou = localization_utils.IOU(personBox, saliencyBox)
    #                             # tmp_rgion = localization_utils.joinBBoxes(saliencyBox,personBox) #Nooo si el saliente bbox es todo frame
    #                             tmp_rgion = personBox
    #                             tmp_rgion.score = iou
    #                             temporal_ious_regions.append(tmp_rgion)
    #                             print('----IOU (person and dynamic region): ', str(iou))
    #                             if iou >= iou_threshold:
    #                                 abnormal_region = localization_utils.joinBBoxes(saliencyBox,personBox)
    #                                 abnormal_region.score = iou
    #                                 anomalous_regions.append(abnormal_region)
                            
    #                     if len(anomalous_regions) > 0:
    #                         anomalous_regions.sort(key=lambda x: x.iou, reverse=True) #sort by iuo to get only one anomalous regions
    #                         anomalous_regions[0].score = -1
    #                         bbox_last = anomalous_regions[0]
    #                         break
    #                     else: # 
    #                         temporal_ious_regions.sort(key=lambda x: x.iou, reverse=True)
    #                         anomalous_regions.append(temporal_ious_regions[0])
    #                         break

    #                         # print('anomalous_regions final: ', len(anomalous_regions))
    #                     ######################################              
                        
    #                     # l = [segmentBox]
    #                     # print('=========== Ground Truth Score =============')
    #                     # scoring.getScoresFromRegions(video_name[0], bbox_segments, l, classifier, transforms_dataset['test'])
                    
    #                     # for b in segment_real_info['real_bboxes']:
    #                     #     b.score = l[0].score
    #                 if len(anomalous_regions) > 0 :
    #                     break
    #             else:
    #                 persons_in_segment.append(None) #only to plot

    #         if len(anomalous_regions) == 0:
    #             print('Tracking Algorithm FAIL!!!!. Using last localization...')
    #             anomalous_regions.append(bbox_last)   
            
    #         # video_persons_in_segment.append(persons_in_frame)
    #         # video_persons_segment_filtered.append(persons_segment_filtered)
    #         # video_anomalous_regions.append(anomalous_regions)

    #         iou = localization_utils.IOU(segmentBox,anomalous_regions[0])
    #         row = [video_name[0]+'---segment No: '+str(num_segment), iou]
    #         data_rows.append(row)
    #         # data_rows_video.append(row)
    #         dis_images, segment_info, idx_next_segment = anomalyDataset.computeSegmentDynamicImg(idx_video=idx_video, idx_next_segment=idx_next_segment)
    #         bbox_last = anomalous_regions[0]        
    #         if plot:
    #             dynamic_image = ttttt
    #             # preprocess = preprocesing_outs[3]
    #             saliencia_suave = np.array(real_frames[0])
    #             img_contuors, contours, hierarchy = localization_utils.findContours(preprocesing_outs[0], remove_fathers=True)
    #             for i in range(len(contours)):
    #                 cv2.drawContours(saliencia_suave, contours, i, (0, 0, 255),2, cv2.LINE_8, hierarchy, 0)
    #             # saliencia_suave = (saliencia_suave - (0, 0, 255)) / 2
    #             # saliencia_suave = np.array(real_frames[0])-saliencia_suave

    #             preprocesing_outs[0] = np.stack((preprocesing_outs[0],)*3, axis=-1)
    #             preprocesing_outs[1] = np.stack((preprocesing_outs[1],)*3, axis=-1)
    #             # print('preprocess: ', preprocesing_outs[0].shape, preprocesing_outs[1].shape, preprocesing_outs[2].shape, preprocesing_outs[3].shape)
    #             mascara = np.stack((mascara,)*3, axis=-1)
    #             BORDER = np.ones((h,20,3))
                
    #             preprocess = np.concatenate((BORDER, mascara,BORDER, preprocesing_outs[0], BORDER, preprocesing_outs[1], BORDER, preprocesing_outs[2], BORDER, preprocesing_outs[3],BORDER), axis=1)
    #             plotOpencv(real_frames,real_bboxes, persons_in_segment, anomalous_regions, dynamic_image, saliencia_suave, preprocess, saliency_bboxes, 350)
    #             ######################################
            


def init(dataset, hockey_path_violence, hockey_path_noviolence, path_learning_curves, path_checkpoints, modelType, ndis, num_workers, data_transforms,
    batch_size, num_epochs, feature_extract, joinType, scheduler_type, device, criterion, folds_number, videoSegmentLength, positionSegment):
    
    for numDiPerVideos in ndis: #for experiments
        train_lost = []
        train_acc = []
        test_lost = []
        test_acc = []
        datasetAll, labelsAll, numFramesAll = initializeDataset.createDataset(hockey_path_violence, hockey_path_noviolence)  #shuffle
            
        df = pd.DataFrame(list(zip(datasetAll, labelsAll, numFramesAll)), columns=['video', 'label', 'numFrames'])
        export_csv = df.to_csv ('hockeyFigths.csv')
            # combined = list(zip(datasetAll, labelsAll, numFramesAll))
            # random.shuffle(combined)
            # datasetAll[:], labelsAll[:], numFramesAll[:] = zip(*combined) 
            # train_idx, test_idx = None, None
        # elif dataset == 'violentflows':

        print("CONFIGURATION: ", "modelType:", modelType, ", numDiPerVideos:", numDiPerVideos, ", batch_size:", batch_size, ", num_epochs:",
                num_epochs, ", feature_extract:", feature_extract, ", joinType:", joinType, ", scheduler_type: ", scheduler_type, )

        
        fold = 0
        for train_idx, test_idx in k_folds(n_splits=folds_number, subjects=len(datasetAll)):
        # for dataset_train, dataset_train_labels,dataset_test,dataset_test_labels   in k_folds_from_folders(vif_path, 5):
            fold = fold + 1
            print("**************** Fold:{}/{} ".format(fold, folds_number))
            train_x, train_y, test_x, test_y = None, None, None, None
            print('fold: ',len(train_idx),len(test_idx))
            train_x = list(itemgetter(*train_idx)(datasetAll))
            train_y = list(itemgetter(*train_idx)(labelsAll))
            train_numFrames = list(itemgetter(*train_idx)(numFramesAll))
            test_x = list(itemgetter(*test_idx)(datasetAll))
            test_y = list(itemgetter(*test_idx)(labelsAll))
            test_numFrames = list(itemgetter(*test_idx)(numFramesAll))
            initializeDataset.print_balance(train_y, test_y)
                
            # train_x, train_y, train_numFrames, test_x, test_y, test_numFrames, data_transforms, numDiPerVideos, batch_size, num_workers, videoSegmentLength, positionSegmen
            dataloaders_dict = initializeDataset.getDataLoaders(train_x, train_y, train_numFrames, test_x, test_y, test_numFrames, data_transforms,
                                                 numDiPerVideos, batch_size, num_workers, videoSegmentLength, positionSegment)
            # data_rows = []
            # for inputs, labels, video_names, bbox_segments in dataloaders_dict["train"]:
            #     print('videos names: ', video_names, labels)

                # row = [video_name[0]+'---segment No: '+str(num_segment), iou]
                # data_rows.append(row)
            MODEL_NAME = modelType+'-'+str(numDiPerVideos)+'-'+joinType+'-segmentLength:'+str(videoSegmentLength)+'-positionSegment:'+positionSegment+'-numEpochs:'+str(num_epochs)
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
            # MODEL_NAME = get_model_name(modelType, scheduler_type, numDiPerVideos, feature_extract, joinType)
            # if folds_number == 1:
            #     MODEL_NAME = MODEL_NAME+constants.LABEL_PRODUCTION_MODEL
            # print('model_name: ', MODEL_NAME)

            if folds_number == 1:
                tr = trainer.Trainer(model, dataloaders_dict, criterion, optimizer, exp_lr_scheduler, device, num_epochs, os.path.join(path_checkpoints, MODEL_NAME),numDiPerVideos)
            else:
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
    parser.add_argument("--dataset",type=str)
    parser.add_argument("--vifPath",type=str,default=constants.PATH_VIOLENTFLOWS_FRAMES,help="Directory for Violent Flows dataset")
    parser.add_argument("--pathViolence",type=str,default=constants.PATH_HOCKEY_FRAMES_VIOLENCE,help="Directory containing violence videos")
    parser.add_argument("--pathNonviolence",type=str,default=constants.PATH_HOCKEY_FRAMES_NON_VIOLENCE,help="Directory containing non violence videos")
    parser.add_argument("--pathLearningCurves", type=str, default=constants.PATH_VIOLENCE_LEARNING_CURVES, help="Directory containing results")
    parser.add_argument("--checkpointPath", type=str, default=constants.PATH_VIOLENCE_CHECKPOINTS)
    parser.add_argument("--modelType",type=str,default="alexnet",help="model")
    parser.add_argument("--numEpochs",type=int,default=30)
    parser.add_argument("--batchSize",type=int,default=64)
    parser.add_argument("--featureExtract",type=lambda x: (str(x).lower() == 'true'), default=False, help="to fine tunning")
    parser.add_argument("--schedulerType",type=str,default="OnPlateau",help="learning rate scheduler")
    parser.add_argument("--ndis", nargs='+', type=int, help="num dyn imgs")
    parser.add_argument("--joinType", type=str)
    parser.add_argument("--foldsNumber", type=int, default=5)
    parser.add_argument("--numWorkers", type=int, default=4)
    parser.add_argument("--videoSegmentLength", type=int)
    parser.add_argument("--positionSegment", type=str)

    args = parser.parse_args()
    dataset = args.dataset
    vif_path = args.vifPath
    path_learning_curves = args.pathLearningCurves
    path_violence = args.pathViolence
    path_noviolence = args.pathNonviolence
    modelType = args.modelType
    batch_size = args.batchSize
    num_epochs = args.numEpochs
    feature_extract = args.featureExtract
    joinType = args.joinType
    scheduler_type = args.schedulerType
    ndis = args.ndis
    path_checkpoints = args.checkpointPath
    folds_number = args.foldsNumber
    num_workers = args.numWorkers
    input_size = 224
    videoSegmentLength = args.videoSegmentLength
    positionSegment = args.positionSegment

    transforms = createTransforms(input_size)
    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()
    # path_models = '/media/david/datos/Violence DATA/violentflows/Models/'
    # path_results = '/media/david/datos/Violence DATA/violentflows/Results/'+dataset_source
    # gpath = '/media/david/datos/Violence DATA/violentflows/movies Frames'

    # init(dataset, path_violence, path_noviolence, path_learning_curves, path_checkpoints, modelType, ndis, num_workers, transforms,
    #         batch_size, num_epochs, feature_extract, joinType, scheduler_type, device, criterion, folds_number,videoSegmentLength, positionSegment)
    num_classes = 2
    saliency_model_file = 'd'
    input_size = (224,224)
    saliency_model_file = 'SALIENCY/Models/anomaly/mask_model_10_frames_di__epochs-12.tar'
    threshold = 0.5
    saliency_tester = saliencyTester.SaliencyTester(saliency_model_file, num_classes, None, test_names,
                                        input_size, saliency_model_config, ndis, None)
    typePersonDetector = 'yolov3'
    only_video_name = None
    plot = True
    h = 288
    w = 360
    overlapping = 0.5
    datasetAll, labelsAll, numFramesAll = initializeDataset.createDataset(path_violence, path_noviolence, suffle=False)
    dataloader, violenceDataset = getOnlineDataLoader(datasetAll, labelsAll, numFramesAll, transform, numDiPerVideos, batch_size, num_workers, overlapping)

    online(violenceDataset, saliency_tester, type_person_detector, h, w, plot, only_video_name)

__main__()

