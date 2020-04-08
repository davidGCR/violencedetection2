
import sys
# import include
sys.path.insert(1,'/Users/davidchoqueluqueroman/Desktop/PAPERS-CODIGOS/violencedetection2')
# sys.path.insert(1, '/media/david/datos/PAPERS-SOURCE_CODE/violencedetection')
# from include import *
import argparse
import ANOMALYCRIME.transforms_anomaly as transforms_anomaly
import ANOMALYCRIME.anomalyInitializeDataset as anomalyInitializeDataset
# import SALIENCY.saliencyTester as saliencyTester

# from saliencyTester import *
# from SALIENCY.saliencyModel  import SaliencyModel
import SALIENCY.saliencyTester as saliencyTester
import constants
import torch
import os
import tkinter
from PIL import Image, ImageFont, ImageDraw, ImageTk
import numpy as np
import cv2
import glob
from localization_utils import tensor2numpy
import localization_utils
from point import Point
from bounding_box import BoundingBox
import matplotlib.pyplot as plt
from YOLOv3 import yolo_inference
import torchvision.transforms as transforms
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import pandas as pd
import torchvision
import MaskRCNN
from torchvision.utils import make_grid
import scoring
from torch.utils.data._utils.collate import default_collate
from tester import Tester
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score, auc
import ANOMALYCRIME.datasetUtils as datasetUtils
import time
from FPS import FPSMeter
from util import saveList2
import pickle
import plot as PLOT

def maskRCNN():
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

def f(image):
        return np.array(image)

def paste_imgs_on_axes(images, axs):
    ims = []    
    r = axs.shape[0]-1
    for idx in range(len(images)):
        img, title = images[idx]
        imag = axs[r, idx].imshow(f(img))
        
        axs[r, idx].set_title(title)
        ims.append(imag)
    return ims

def pytorch_show(img):
    # img = img / 2 + 0.5 
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')

def myplot(r, c, font_size, grid_static_imgs, img_source, real_frames, real_bboxes, saliency_bboxes, persons_in_segment,
            persons_segment_filtered, anomalous_regions):
    # create a figure with two subplots
    fig, axes = plt.subplots(r, c, figsize=(20, 10))
    plt.subplots_adjust(hspace=0.5, wspace=0, left = 0.03, right = 0.99)

    for i in range(grid_static_imgs.shape[0]-1):
        for j in range(grid_static_imgs.shape[1]):
            # print(type(grid_static_imgs[i,j]), i,j)
            im1 = axes[i, j].imshow(grid_static_imgs[i,j][0])  
            axes[i, j].set_title(grid_static_imgs[i,j][1])
    
    
    img_source = localization_utils.plotOnlyBBoxOnImage(img_source, saliency_bboxes, constants.PIL_YELLOW)
    # img_source = localization_utils.setLabelInImage(img_source,saliency_bboxes, 'saliency','yellow',10,'left_corner','black' )
    image_anomalous, image_anomalous_final = None, None
    ims = []
    for i in range(len(real_frames)):
    #     # real_frame = real_frames[i].copy()
        images = []
        gt_image = localization_utils.plotOnlyBBoxOnImage(real_frames[i].copy(), real_bboxes[i], constants.PIL_RED)

        img_source = localization_utils.plotOnlyBBoxOnImage(img_source, real_bboxes[i], constants.PIL_RED)
        img_source = localization_utils.setLabelInImage(img_source, real_bboxes[i], 'score', constants.PIL_RED, font_size, 'left_corner', 'white')
        img_source = localization_utils.plotOnlyBBoxOnImage(img_source, saliency_bboxes, constants.PIL_YELLOW)
        for s_box in saliency_bboxes:
            img_source = localization_utils.setLabelInImage(img_source, s_box, 'score' ,font_color=constants.PIL_YELLOW,font_size=font_size,pos_text='right_corner',background_color='black' )
        
        
        images.append((img_source,'source image'))
        
        if len(saliency_bboxes) > 0:
            image_saliency = localization_utils.plotOnlyBBoxOnImage(gt_image.copy(), real_bboxes[i], constants.PIL_RED)
            image_saliency = localization_utils.setLabelInImage(image_saliency, real_bboxes[i], 'score', constants.PIL_RED, font_size, 'left_corner', 'black')
            
            image_saliency = localization_utils.plotOnlyBBoxOnImage(image_saliency, saliency_bboxes, constants.PIL_YELLOW)
            for s_box in saliency_bboxes:
                image_saliency = localization_utils.setLabelInImage(image_saliency,s_box, 'score' ,constants.PIL_YELLOW,font_size,'right_corner','black' )
            images.append((image_saliency,'gt image with bboxes'))
            
            image_persons = image_saliency.copy()
            for person in persons_in_segment:
                image_persons = localization_utils.plotOnlyBBoxOnImage(image_persons, person, constants.PIL_GREEN)
            # image_persons = localization_utils.setLabelInImage(image_persons,persons_in_segment[i], text='person',font_color='black',font_size=font_size,pos_text='left_corner',background_color='white' )
            images.append((image_persons,'persons'))

            image_persons_filt = localization_utils.plotOnlyBBoxOnImage(real_frames[i].copy(), persons_segment_filtered[0], constants.PIL_MAGENTA)
            # image_persons_filt = localization_utils.setLabelInImage(image_persons_filt, persons_segment_filtered[i], text='person_filter', font_color=constants.PIL_MAGENTA, font_size=font_size, pos_text='left_corner', background_color='white')
            for k,ar in enumerate(anomalous_regions):
                image_persons_filt = localization_utils.plotOnlyBBoxOnImage(image_persons_filt, anomalous_regions[k], constants.PIL_YELLOW)
                image_persons_filt = localization_utils.setLabelInImage(image_persons_filt,ar, 'score' ,constants.PIL_WHITE,font_size,'right_corner','black' )
            images.append((image_persons_filt,'persons close and anomalous regions'))
            
            image_anomalous = localization_utils.plotOnlyBBoxOnImage(real_frames[i].copy(), anomalous_regions[0], constants.PIL_BLUE)
            segmentBox = localization_utils.getSegmentBBox(real_bboxes)
            image_anomalous = localization_utils.plotOnlyBBoxOnImage(image_anomalous, segmentBox, constants.PIL_RED)
            images.append((image_anomalous,'abnormal regions'))                    
        
        
        ims.append(paste_imgs_on_axes(images, axes))
    # print('ims: ', len(ims))
    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=100)
    # ani.save('RESULTS/animations/animation.mp4', writer=writer)          
    plt.show()

def getPersonDetectorModel(detector_type, device):
    classes = None
    if detector_type == constants.YOLO:
        img_size = 416
        weights_path = "YOLOv3/weights/yolov3.weights"
        class_path = "YOLOv3/data/coco.names"
        model_def = "YOLOv3/config/yolov3.cfg"
        person_model, classes = yolo_inference.initializeYoloV3(img_size, class_path, model_def, weights_path, device)
        
        # print('persons_in_frame MaskRCNN: ', len(persons_in_frame))
    elif detector_type == constants.MASKRCNN:
        person_model = maskRCNN()
        

    return person_model, classes


def plotOpencv(video_name, no_segment, prediction, images,gt_bboxes, persons_in_segment, bbox_predictions, dynamic_image, mascara, preprocess, saliency_bboxes, sep, delay):
    red = (0, 0, 255)
    green = (0, 255, 0)
    blue = (255, 0, 0)
    yellow = (0, 255, 255)
    magenta = (255,50,255)
    for idx,image in enumerate(images):
        
        image = np.array(image)
        w = image.shape[0]
        h = image.shape[1]
        if prediction == 1:
            cv2.rectangle(image, (0, 0), (h, w), magenta, 4)

        image_saliencies = image.copy()
        image_persons = image.copy()
        image_persons_close = image.copy()
        image_anomalous = image.copy()
        # image_contours = image.copy()

        for gt_bbox in gt_bboxes:
            # print('ccooooooooooocluded: ', gt_bbox.occluded)
            if gt_bbox.occluded == 0:
                cv2.rectangle(image, (int(gt_bbox.pmin.x), int(gt_bbox.pmin.y)), (int(gt_bbox.pmax.x), int(gt_bbox.pmax.y)), red, 2)
        
        
        for s_bbox in saliency_bboxes:
            cv2.rectangle(image_saliencies, (int(s_bbox.pmin.x), int(s_bbox.pmin.y)), (int(s_bbox.pmax.x), int(s_bbox.pmax.y)), yellow, 1)
        
        if idx < len(persons_in_segment):
            if persons_in_segment[idx] is not  None:
                for person in persons_in_segment[idx]:
                    cv2.rectangle(image_persons, (int(person.pmin.x), int(person.pmin.y)), (int(person.pmax.x), int(person.pmax.y)), green, 1)
        
        for pred in bbox_predictions:
            if pred is not None:
                cv2.rectangle(image_anomalous, (int(pred.pmin.x), int(pred.pmin.y)), (int(pred.pmax.x), int(pred.pmax.y)), blue, 2)
            vid_path = os.path.join('RESULTS/animations', video_name)
            if not os.path.exists(vid_path):
                os.makedirs(vid_path)
            status = cv2.imwrite(vid_path+'/'+str(no_segment)+'.png',image_anomalous)

        
        # aa = np.concatenate((image, image), axis=1)
        cv2.imshow('image', image)
        cv2.imshow('dynamic_image', dynamic_image)
        cv2.imshow('image_saliencies', image_saliencies)
        cv2.imshow('image_persons', image_persons)
        cv2.imshow('image_anomalous', image_anomalous)
        cv2.imshow('mascara', mascara)
        cv2.imshow('preprocess', preprocess)

        pos_x = 20
        cv2.namedWindow("image");#x,y
        cv2.moveWindow("image", pos_x, 100);

        cv2.namedWindow("dynamic_image");
        cv2.moveWindow("dynamic_image", pos_x+sep, 100);
        
        cv2.namedWindow("image_saliencies");
        cv2.moveWindow("image_saliencies", pos_x+2*sep, 100);

        cv2.namedWindow("image_persons");
        cv2.moveWindow("image_persons", pos_x+3*sep, 100);

        cv2.namedWindow("image_anomalous");
        cv2.moveWindow("image_anomalous", pos_x+4*sep, 100);

        cv2.namedWindow("mascara");
        cv2.moveWindow("mascara", pos_x+5*sep, 100);

        cv2.namedWindow("preprocess");
        cv2.moveWindow("preprocess", 20,500);
        # k = cv2.waitKey(0)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break   
        # if k == 27:         # wait for ESC key to exit
        #     cv2.destroyAllWindows()

def temporalTest(anomalyDataset, model_name, config, class_tester, saliency_tester, type_person_detector, h, w, plot, only_video_name, delay, datasetType='UCF2Local'):
    # person_model, classes = getPersonDetectorModel(type_person_detector)
    data_rows = []
    indx_flac = -1
    num_video_segments = 0
    videos_scores = []
    total_tp = 0
    total_fp = 0
    num_pos_frame = 0
    num_neg_frame = 0

    y_truth = []
    y_pred = []
    skip = 1

    for idx_video, data in enumerate(anomalyDataset):
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
        last_score = 0
       
        video_name, label = data
        print("-" * 150, 'video No: ', idx_video, ' name: ', video_name)
        video_name = [video_name]
        label = torch.tensor(label)
        block_dinamyc_images, idx_next_block, block_gt, s_time = anomalyDataset.computeBlockDynamicImg(idx_video, idx_next_block=0, skip=skip)
        print('block_dinamyc_images: ', block_dinamyc_images.size())
        # dis_images, segment_info, idx_next_segment = anomalyDataset.computeSegmentDynamicImg(idx_video=idx_video, idx_next_segment=0)
        num_block = 0
        video_y_pred = []
        video_y_pred_class = []
        video_y_gt = []
        # numFrames = anomalyDataset.getNumFrames(idx_video)
        while (block_dinamyc_images is not None): #dis_images : torch.Size([3, 224, 224])
            # print('Dynamic image block: ', block_dinamyc_images.size()) #torch.Size([1, 3, 224, 224])
            # dynamic_img = torch.unsqueeze(dis_images, dim=0)
            # print('---numBlock: ', num_block, '--idx_next_block:', idx_next_block)
            prediction, score = class_tester.predict(block_dinamyc_images)
            
            # y_block_pred = localization_utils.countTruePositiveFalsePositive(block_boxes_info, prediction, score, 0.8)
            y_block_pred = []
            y_block_pred_class = []
            for info in block_gt:
                y_block_pred.append(score)
                y_block_pred_class.append(prediction)
            y_pred.extend(y_block_pred)
            video_y_pred.extend(y_block_pred)
            video_y_pred_class.extend(y_block_pred_class)
            # for info in block_boxes_info:
            #     video_y_pred.append(score)
            # video_y_pred.extend(y_block_pred)
            # y_pred_scored_based.extend(y_score_based)
            # if datasetType == 'waqas':
            #     p, n, y_block_truth = localization_utils.countTemporalGroundTruth(block_boxes_info)
            # else:
            y_block_truth = []
            for info in block_gt:
                if info[1]== 0:
                    y_block_truth.append(1)
                else:
                    y_block_truth.append(0)
            y_truth.extend(y_block_truth)
            video_y_gt.extend(y_block_truth)
            # print(y_block_truth)
            # p, n, y_block_truth = localization_utils.countPositiveFramesNegativeFrames(block_boxes_info)
            
            # video_y_gt.extend(y_block_truth)
            # num_pos_frame += p
            # num_neg_frame += n
            # total_fp += fp
            # total_tp += tp
            num_block += 1
            if plot:
                # print('prediction: ', prediction, score)
                # if datasetType == 'waqas':
                # else:
                # frames_names, real_frames, real_bboxes = localization_utils.getFramesFromBlock(video_name[0], block_gt)
                # real_frames = []
                # for 


                # block_dinamyc_images = torch.unsqueeze(block_dinamyc_images, 0)
                dynamic_image_plot = saliency_tester.min_max_normalize_tensor(block_dinamyc_images)
                dyImgs = []
                for di in dynamic_image_plot:
                    di = di.numpy().transpose(1, 2, 0)
                    di = cv2.resize(di, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
                    dyImgs.append(di)
                dynamic_image_plot = np.concatenate(np.array(dyImgs), axis=1)
                # dynamic_image_plot = dynamic_image_plot.numpy()[0].transpose(1, 2, 0)
                # dynamic_image_plot = cv2.resize(dynamic_image_plot, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(dynamic_image_plot,str(score),(0,25), font, 0.6,constants.yellow,2,cv2.LINE_AA)
                # plt.imshow(dynamic_image_plot)
                # plt.show()
                

                pos_x = 20
                sep = 400

                # print('ground-truth checking: ', len(real_frames), len(y_block_truth))
                for i, frame_gt in enumerate(block_gt):
                    frame_path = os.path.join(video_name[0],frame_gt[0])
                    frame = Image.open(frame_path)
                    frame = np.array(frame)
                    occluded = int(frame_gt[1])
                    gt_bbox = BoundingBox(Point(float(frame_gt[constants.IDX_XMIN]), float(frame_gt[constants.IDX_YMIN])),
                            Point(float(frame_gt[constants.IDX_XMAX]), float(frame_gt[constants.IDX_YMAX])), occluded=occluded)
                    # font = cv2.FONT_HERSHEY_SIMPLEX
                    # cv2.putText(frame,str(score),(0,25), font, 0.6,constants.red,2,cv2.LINE_AA)
                    cv2.putText(frame,str(frame_gt[constants.IDX_NUMFRAME]),(15,40), font, 0.6,constants.magenta,2,cv2.LINE_AA)
                    wn = frame.shape[0]
                    hn = frame.shape[1]
                    if prediction == 1:
                        cv2.rectangle(frame, (10, 10), (hn-10, wn-10), constants.magenta, 4)
                    if y_block_truth[i] == 1:
                        cv2.rectangle(frame, (0, 0), (hn, wn), constants.yellow, 4)

                    # gt_bbox = real_bboxes[i]
                    if gt_bbox.occluded == 0:
                        cv2.rectangle(frame, (int(gt_bbox.pmin.x), int(gt_bbox.pmin.y)), (int(gt_bbox.pmax.x), int(gt_bbox.pmax.y)), constants.red, 2)
                    cv2.imshow('frame', frame)
                    cv2.imshow('dynamic_image', dynamic_image_plot)
                    cv2.namedWindow("frame");#x,y
                    cv2.moveWindow("frame", pos_x, 100);
                    cv2.namedWindow("dynamic_image");
                    cv2.moveWindow("dynamic_image", pos_x+sep, 100);
                    if cv2.waitKey(delay) & 0xFF == ord('q'):
                        break
            
            block_dinamyc_images, idx_next_block, block_gt, s_time = anomalyDataset.computeBlockDynamicImg(idx_video, idx_next_block=idx_next_block, skip=skip)
            # print(block_dinamyc_images.size())
        v_folder, v_name = os.path.split(video_name[0])
        video_result = {'name': v_name,
                        'y_truth': video_y_gt,
                        'y_pred_score': video_y_pred,
                        'y_pred_class': video_y_pred_class}
        # print(video_y_pred_class)
        if plot:
            PLOT.plot_temporal_results(video_result,threshold=0.5,save=False)
        else:
            # print('gererere')
            PLOT.plot_temporal_results(video_result,threshold=0.5,save=True)
            pickle.dump(video_result, open(os.path.join(constants.PATH_VIOLENCE_TMP_RESULTS,v_name+'.pkl'),"wb"))
        # vid_auc = roc_auc_score(video_y_gt, video_y_pred)
        # print('Video: ', video_name[0],str(vid_auc))

    # fpr, tpr, _ = roc_curve(y_truth, y_pred_scored_based)
    y_truth = np.array(y_truth)
    y_pred = np.array(y_pred)
    y_pred_cpy = -y_pred
    idxs = y_pred_cpy.argsort()
    y_truth = y_truth[idxs]
    y_pred = y_pred[idxs]

    # print(y_pred)
    
    
    fpr, tpr, _ = roc_curve(y_truth, y_pred)
    if not plot:
        #  config = {'videoBlockLength': 70,
        #         'BlockOverlap':0,
        #         'videoSegmentLength': 20,
        #         'SegmentOverlap': 0.5,
        #         'model': 'testresnet50-6-Finetuned:True-maxTempPool-numEpochs:9-videoSegmentLength:20-overlaping:0.5-only_violence:True',
        #         'tpr': tpr,
        #         'fpr': fpr}
        config.update({'y_truth': y_truth,'y_pred': y_pred})
        config.update({'tpr': tpr, 'fpr': fpr})
        pickle.dump(config, open(os.path.join(constants.PATH_VIOLENCE_ROC_CURVES,'Model:%s-BLength:%d-BOverlp:%.2f-SLength:%d-SOverlp:%.2f-BlockNumDynImgs:%d.pkl'%(str(model_name),config['videoBlockLength'],config['BlockOverlap'],config['videoSegmentLength'],config['SegmentOverlap'],config['numDynamicImgsPerBlock'])), "wb"))
        # saveList2(os.path.join(constants.PATH_VIOLENCE_ROC_CURVES,model_name[:-4] + '-fpr.txt'), fpr)
        # saveList2(os.path.join(constants.PATH_VIOLENCE_ROC_CURVES,model_name[:-4] + '-tpr.txt'), tpr)
          
    PLOT.plotROCCurve(tpr,fpr)
    # fig = plt.figure()
    # ax=fig.gca()
    # vauc = auc(fpr, tpr)
    # plt.plot(fpr, tpr, lw=1, color='red', marker='.', label='(AUC = %0.4f)' % (vauc))
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # plt.xlim(xmin=0.0, xmax=1)
    # plt.ylim(ymin=0.0, ymax=1)
    # ax.set_xticks(np.arange(0,1,0.1))
    # ax.set_yticks(np.arange(0,1,0.1))
    # # plt.scatter()
    # plt.grid()
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC')
    # plt.legend(loc="lower right")
    # plt.show()

    # lr_auc = roc_auc_score(y_truth, y_pred) #type 1
    # vauc = auc(fpr, tpr) #type 2
    # print('AUC: ', str(lr_auc), vauc)
    # plt.plot(fpr, tpr, color='darkorange', marker='.', label='Curva ROC (area = %0.4f)' % vauc)
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic')
    # plt.legend(loc="lower right")
    # plt.show()

    # num_positive_pred = len([i for i in y_pred if i >= 0.5])
    # false_alarm_rate = num_positive_pred / len(y_pred)
    # print('False alarm rate: ', str(false_alarm_rate))
    # print('num positive: ', str(num_positive_pred))
    # print(y_truth)

        # show the legend
        # pyplot.legend()
        # show the plot
    # 
    # print('Frames processed: ', len(y_truth), len(y_pred))
    # 

def localization(persons_in_frame, thresh_close_persons, saliency_bboxes, iou_threshold = 0.3):
    persons_filtered, only_joined_regions = localization_utils.filterClosePersonsInFrame(persons_in_frame, thresh_close_persons)
    temporal_ious_regions = []
    anomalous_regions = []
    for personBox in persons_filtered:
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
        
    if len(anomalous_regions) > 1:
        anomalous_regions.sort(key=lambda x: x.iou, reverse=True) #sort by iuo to get only one anomalous regions
        anomalous_regions[0].score = -1
        # bbox_last = anomalous_regions[0]
    #     break
    # else: # 
    #     temporal_ious_regions.sort(key=lambda x: x.iou, reverse=True)
    #     anomalous_regions.append(temporal_ious_regions[0])
    #     break
    return anomalous_regions

def spatioTemporalDetection(anomalyDataset, class_tester, saliency_tester, type_person_detector, h, w, plot, only_video_name, delay, device):
    person_model, classes = getPersonDetectorModel(type_person_detector, device)
    # person_model = person_model.to(device)
    # print('model cuda: ', next(person_model.parameters()).is_cuda)
    bbox_last = None
    distance_th = 35.5
    thres_intersec_lastbbox_current = 0.4
    ious = []
    times = []
    indx_flac = -1
    num_video_segments = 0
    videos_scores = []
    total_tp = 0
    total_fp = 0
    num_pos_frame = 0
    num_neg_frame = 0
    y_truth = []
    y_pred = []
    num_iter = 1
    
    # fpsMeterMask = FPSMeter()
    # fpsMeterRefinement = FPSMeter()
    pos_x = 20
    sep = 400

    for idx_video, data in enumerate(anomalyDataset):

        fpsMeterMask = FPSMeter()
        fpsMeterRefinement = FPSMeter()

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
        # di_spend_times = []
        bbox_last = None
        print("-" * 150, 'video No: ', idx_video)
        video_name, label = data
        video_name = [video_name]
        label = torch.tensor(label)

        ################### BLOCK ####################
        # dinamycImages, idx_next_block, block_gt, spend_time
        block_dinamyc_images, idx_next_block, block_gt, spend_time = anomalyDataset.computeBlockDynamicImg(idx_video, idx_next_block=0, skip=1)
       
        # prediction, score = class_tester.predict(block_dinamyc_images)
        # # print('Score: ', score)
        # tp, fp, y_block_pred, y_score_based = localization_utils.countTruePositiveFalsePositive(block_boxes_info, prediction, score, threshold=0)
        # p, n, y_block_truth = localization_utils.countPositiveFramesNegativeFrames(block_boxes_info)
        # y_truth.extend(y_block_truth)
        # y_pred.extend(y_block_pred)
        # num_pos_frame += p
        # num_neg_frame += n
        # total_fp += fp
        # total_tp += tp
        # dis_images, segment_info, idx_next_segment, spend_time_dyImg = anomalyDataset.computeSegmentDynamicImg(idx_video=idx_video, idx_next_segment=0)
        
        # tp, fp, y_block_pred, y_score_based = localization_utils.countTruePositiveFalsePositive(segment_info, prediction, score, threshold=0)
        
        
        
        num_block = 0
        while (block_dinamyc_images is not None): #dis_images : torch.Size([3, 224, 224])
            num_block += 1            
            print(video_name, '---segment No: ', str(num_block),'-----', block_dinamyc_images.size(), block_dinamyc_images.is_cuda)
            
            prediction, score = class_tester.predict(block_dinamyc_images)
            # print('- dis_images: ', type(dis_images), dis_images.size())
            # print('- video_name: ', type(video_name), video_name)
            # print('- label: ', type(label))
            # print('- segment_info: ', type(segment_info), segment_info[0])
            # - dis_images:  <class 'torch.Tensor'>
            # - video_name:  <class 'str'>
            # - labels:  <class 'int'>
            # - bbox_segments:  <class 'list'>  [['frame645.jpg', 0, 72, 67, 133, 148],..., []
            # block_dinamyc_images = torch.unsqueeze(block_dinamyc_images, 0)
            # block_boxes_info = default_collate([block_boxes_info])
            # block_boxes_info = np.array(block_boxes_info)

            block_dinamyc_images = block_dinamyc_images / 2 + 0.5 
            ttttt = saliency_tester.min_max_normalize_tensor(block_dinamyc_images) 
            ttttt = ttttt.numpy()[0].transpose(1, 2, 0)
            ttttt = cv2.resize(ttttt, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
            
            ########################### LOCALIZATION ##################################################3
            ######################## mask
            masks, mask_time = saliency_tester.compute_mask(block_dinamyc_images, label)  #[1, 1, 224, 224]
            fpsMeterMask.update(mask_time)
            
            mascara = masks.detach().cpu()#(1, 1, 224, 224)
            mascara = torch.squeeze(mascara,0)#(1, 224, 224)
            mascara = saliency_tester.min_max_normalize_tensor(mascara)  #to 0-1
            mascara = mascara.numpy().transpose(1, 2, 0)

            # ##resize 
            mascara = cv2.resize(mascara, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
            # mascara = dy_gray
            preproc_start_time = time.time()
            saliency_bboxes, preprocesing_outs, contours, hierarchy = localization_utils.computeBoundingBoxFromMask(mascara)  #only one by segment
            saliency_bboxes = localization_utils.removeInsideSmallAreas(saliency_bboxes)  
            preproc_end_time = time.time()
            preproc_time = preproc_end_time-preproc_start_time
            # if plot:
            #     cv2.imshow('Dynamic Image', ttttt)
            #     if cv2.waitKey(delay) & 0xFF == ord('q'):
            #         break
            
            #metricas (no decir es mejor... 2 puntos porcentuales del valor-outperform) en esa base de datos- grafico explicar ejes...porque de los resultados
            #presentar errores

            ################################################## REFINEMENT ################################################################
            # track_start_time = time.time() # Tracking....
            # if bbox_last is not None:
            #     bboxes_with_high_core = []
            #     for bbox in saliency_bboxes:
            #         pred_distance = localization_utils.distance(bbox.center, bbox_last.center)
            #         if pred_distance <= distance_th or bbox.percentajeArea(localization_utils.intersetionArea(bbox,bbox_last)) >= thres_intersec_lastbbox_current:
            #             bbox.score = pred_distance
            #             bboxes_with_high_core.append(bbox)
            #     if len(bboxes_with_high_core) > 0:
            #         bboxes_with_high_core.sort(key=lambda x: x.score)
            #         saliency_bboxes = bboxes_with_high_core
            #     else:
            #         saliency_bboxes = [bbox_last]

            # if len(saliency_bboxes) == 0:
            #     saliency_bboxes = [BoundingBox(Point(0,0),Point(w,h))]
            
            # track_end_time = time.time()
            # track_time = track_end_time - track_start_time
            
            # frames_names, real_frames, real_bboxes = localization_utils.getFramesFromSegment(video_name[0], segment_info, 'all')
            real_frames = []
            real_bboxes = []
            for i, frame_gt in enumerate(block_gt):
                frame_path = os.path.join(video_name[0],frame_gt[0])
                frame = Image.open(frame_path)
                frame = np.array(frame)
                real_frames.append(frame)
                occluded = int(frame_gt[1])
                gt_bbox = BoundingBox(Point(float(frame_gt[constants.IDX_XMIN]), float(frame_gt[constants.IDX_YMIN])),
                        Point(float(frame_gt[constants.IDX_XMAX]), float(frame_gt[constants.IDX_YMAX])), occluded=occluded)
                real_bboxes.append(gt_bbox)
                # font = cv2.FONT_HERSHEY_SIMPLEX
                # cv2.putText(frame,str(score),(0,25), font, 0.6,constants.red,2,cv2.LINE_AA)
                # cv2.putText(frame,str(frame_gt[constants.IDX_NUMFRAME]),(15,40), font, 0.6,constants.magenta,2,cv2.LINE_AA)
                wn = frame.shape[0]
                hn = frame.shape[1]
                # if prediction == 1:
                #     cv2.rectangle(frame, (10, 10), (hn-10, wn-10), constants.magenta, 4)
                # if y_block_truth[i] == 1:
                #     cv2.rectangle(frame, (0, 0), (hn, wn), constants.yellow, 4)

                # gt_bbox = real_bboxes[i]
                if gt_bbox.occluded == 0:
                    cv2.rectangle(frame, (int(gt_bbox.pmin.x), int(gt_bbox.pmin.y)), (int(gt_bbox.pmax.x), int(gt_bbox.pmax.y)), constants.red, 2)
                cv2.imshow('frame', frame)
                dynamic_image_plot = ttttt
                cv2.imshow('dynamic_image', dynamic_image_plot)
                cv2.namedWindow("frame");#x,y
                cv2.moveWindow("frame", pos_x, 100);
                cv2.namedWindow("dynamic_image");
                cv2.moveWindow("dynamic_image", pos_x+sep, 100);
                if cv2.waitKey(delay) & 0xFF == ord('q'):
                    break

            persons_in_segment = []
            persons_segment_filtered = []
            anomalous_regions = []  # to plot
            segment_gt_Box = localization_utils.getSegmentBBox(real_bboxes)
            
            temporal_ious_regions = []
            for idx, frame in enumerate(real_frames): #Detect persons frame by frame 
                # print('Shot frame to process: ',idx)
                persons_in_frame = []
                persons_filtered = []
                # frame = frame.to(device)
                if type_person_detector == constants.YOLO:
                    img_size = 416
                    conf_thres = 0.8
                    nms_thres = 0.4
                    persons_in_frame, detection_time = localization_utils.personDetectionInFrameYolo(person_model, img_size, conf_thres,
                                                                                    nms_thres, classes, frame, device)
                elif type_person_detector == constants.MASKRCNN:
                    mask_rcnn_threshold = 0.4
                    persons_in_frame, detection_time = MaskRCNN.personDetectionInFrameMaskRCNN(person_model, frame, mask_rcnn_threshold)
                # print('--num Persons in frame: ', len(persons_in_frame))
                if len(persons_in_frame) > 0:
                    ref_start_time = time.time()
                    persons_in_segment.append(persons_in_frame)
                    thres_intersec_person_saliency = 0.5
                    thresh_close_persons = 20
                    anomalous_regions = localization(persons_in_frame, thresh_close_persons, saliency_bboxes, thres_intersec_person_saliency)
                    ref_end_time = time.time()
                    refinement_time = ref_end_time - ref_start_time
                    fpsMeterRefinement.update(preproc_time+refinement_time)
                    if len(anomalous_regions) > 0:
                        break
                else:
                    persons_in_segment.append(None) #only to plot

            block_dinamyc_images, idx_next_block, block_boxes_info, spend_time = anomalyDataset.computeBlockDynamicImg(idx_video, idx_next_block=idx_next_block, skip=1)   
        
            # iou = localization_utils.IOU(segment_gt_Box,anomalous_regions[0])
            # segment_iou = [video_name[0]+'-segment No: '+str(num_segment), iou]
            # ious.append(segment_iou)s
                
            if plot:
                dynamic_image = ttttt
                # preprocess = preprocesing_outs[3]
                saliencia_suave = np.array(real_frames[0])
                img_contuors, contours, hierarchy = localization_utils.findContours(preprocesing_outs[0], remove_fathers=True)
                for i in range(len(contours)):
                    cv2.drawContours(saliencia_suave, contours, i, (0, 0, 255),2, cv2.LINE_8, hierarchy, 0)
                # saliencia_suave = (saliencia_suave - (0, 0, 255)) / 2
                # saliencia_suave = np.array(real_frames[0])-saliencia_suave

                preprocesing_outs[0] = np.stack((preprocesing_outs[0],)*3, axis=-1)
                preprocesing_outs[1] = np.stack((preprocesing_outs[1],)*3, axis=-1)
                # print('preprocess: ', preprocesing_outs[0].shape, preprocesing_outs[1].shape, preprocesing_outs[2].shape, preprocesing_outs[3].shape)
                mascara = np.stack((mascara,)*3, axis=-1)
                BORDER = np.ones((h,20,3))
                
                preprocess = np.concatenate((BORDER, mascara,BORDER, preprocesing_outs[0], BORDER, preprocesing_outs[1], BORDER, preprocesing_outs[2], BORDER, preprocesing_outs[3],BORDER), axis=1)
                prediction = 0
                head, tail = os.path.split(video_name[0])
                plotOpencv(tail, num_segment, prediction, real_frames,real_bboxes, persons_in_segment, anomalous_regions, dynamic_image, saliencia_suave, preprocess, saliency_bboxes, 300, delay)
            #     ######################################
            ################################################## REFINEMENT ################################################################
            
        fpsMeterMask.print_statistics()
        fpsMeterRefinement.print_statistics()
        # row_time = [video_name[0], round(fpsMeter.fps(),2), round(fpsMeter.mspf(),2)]
        # print(row)
        # times.append(row_time)

    #################### Localization Error #####################
    # df = pd.DataFrame(ious, columns=['path', 'iou'])
    # seriesObj = df.apply(lambda x: True if x['iou'] <0.5 else False , axis=1)
    # numOfRows = len(seriesObj[seriesObj == True].index)
    # localization_error = numOfRows/len(ious)
    # print('Localization Error: ', str(localization_error))

    ################### ROC - AUC ############################
    # if mode == 'rocauc':
    #     fpr, tpr, _ = roc_curve(y_truth, y_pred)
    #     lr_auc = roc_auc_score(y_truth, y_pred)
    #     plt.plot(fpr, tpr, marker='.', label='test')

    #     plt.xlabel('False Positive Rate')
    #     plt.ylabel('True Positive Rate')
    #     plt.show()
    #     print('AUC: ', str(lr_auc))

def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saliencyModelFile", type=str)
    parser.add_argument("--batchSize", type=int, default=3)
    # parser.add_argument("--numEpochs", type=int, default=10)
    parser.add_argument("--numWorkers", type=int, default=1)
    parser.add_argument("--numDynamicImgsPerBlock", type=int, default=5)
    parser.add_argument("--shuffle", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--plot", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--videoSegmentLength", type=int, default=15)
    parser.add_argument("--videoBlockLength", type=int)
    parser.add_argument("--personDetector", type=str, default=constants.YOLO)
    parser.add_argument("--positionSegment", type=str)
    parser.add_argument("--overlappingBlock", type=float)
    parser.add_argument("--overlappingSegment", type=float)
    parser.add_argument("--videoName", type=str, default=None)
    parser.add_argument("--delay", type=int, default=1)

    args = parser.parse_args()
    plot = args.plot
    maxNumFramesOnVideo = 0
    videoSegmentLength = args.videoSegmentLength
    videoBlockLength = args.videoBlockLength
    typePersonDetector = args.personDetector
    numDynamicImgsPerBlock = args.numDynamicImgsPerBlock
    positionSegment = args.positionSegment
    overlappingSegment = args.overlappingSegment
    overlappingBlock = args.overlappingBlock
    num_classes = 2 #anomalus or not
    input_size = (224,224)
    transforms_dataset = transforms_anomaly.createTransforms(input_size)
    batch_size = args.batchSize
    num_workers = args.numWorkers
    saliency_model_file = args.saliencyModelFile
    shuffle = args.shuffle
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    threshold = 0.5
    video_name = args.videoName
    delay = args.delay

    saliency_model_config = saliency_model_file
    # path_dataset = constants.PATH_UCFCRIME2LOCAL_FRAMES_REDUCED
    path_dataset = constants.PATH_UCFCRIME2LOCAL_FRAMES
    train_videos_path = os.path.join(constants.PATH_UCFCRIME2LOCAL_README, 'Train_split_AD.txt')
    test_videos_path = os.path.join(constants.PATH_UCFCRIME2LOCAL_README, 'Test_split_AD.txt')
# path_dataset, test_videos_path, batch_size, num_workers, videoBlockLength,
#                     numDynamicImgsPerBlock, transform, videoSegmentLength, shuffle,overlappingBlock, overlappingSegment
    # videoBlockLength = 30
    only_anomalous = False
    dataloader_test, test_names, test_labels, anomalyDataset = anomalyInitializeDataset.initialize_test_anomaly_dataset(path_dataset,
                                                        test_videos_path, batch_size, num_workers, videoBlockLength, numDynamicImgsPerBlock,
                                                        transforms_dataset['test'],
                                                        videoSegmentLength, shuffle, overlappingBlock, overlappingSegment, only_anomalous=only_anomalous, only_violence=True)
    
    saliency_tester = saliencyTester.SaliencyTester(saliency_model_file, num_classes, dataloader_test, test_names,
                                        input_size, saliency_model_config, 1, threshold)
    ####Waqas
    # videos_paths, labels, numFrames, tmp_gtruth = datasetUtils.waqas_dataset('/media/david/datos/Violence DATA/AnomalyCRIMEDATASET/waqas/test',
    #                                                                 'waqas/Temporal_Anomaly_Annotation.txt', True)

    # waqas_dataloader, waqas_dataset = anomalyInitializeDataset.waqas_anomaly_downloader(videos_paths,labels, numFrames, batch_size, num_workers, videoBlockLength,
    #                 numDynamicImgsPerBlock, transforms_dataset['test'], videoSegmentLength, shuffle,overlappingBlock, overlappingSegment, tmp_gtruth)
    # # limit = 55
    # # print(videos_paths[0:limit])
    # # print(numFrames[0:limit])
    # # print(labels[0:limit])
    # # print(tmp_gtruth[0:limit])
    # saliency_tester = saliencyTester.SaliencyTester(saliency_model_file, num_classes, waqas_dataloader, videos_paths,
                                        # input_size, saliency_model_config, 1, threshold)
    
    # print(torch.__version__)
    h = 240
    w = 320
    # classifier_file = 'ANOMALYCRIME/checkpoints/resnet50-3-Finetuned:True-maxTempPool-numEpochs:13.pth'
    # classifier_file = 'ANOMALYCRIME/checkpoints/testresnet50-3-Finetuned:True-maxTempPool-numEpochs:24-videoSegmentLength:20-overlaping:0.5-only_violence:True.pth'
    # classifier_file = 'ANOMALYCRIME/checkpoints/testresnet50-3-Finetuned:True-maxTempPool-numEpochs:13-videoSegmentLength:200-overlaping:0.5-only_violence:True.pth'
    # classifier_file = 'ANOMALYCRIME/checkpoints/testresnet50-6-Finetuned:True-maxTempPool-numEpochs:9-videoSegmentLength:20-overlaping:0.5-only_violence:True.pth'
   
    # classifier_file = 'ANOMALYCRIME/checkpoints/testresnet50-3-Finetuned:True-maxTempPool-numEpochs:20-videoSegmentLength:40-overlaping:0.5-only_violence:True.pth'

    classifier_file = 'ANOMALYCRIME/checkpoints/resnet18_Finetuned-False-_di-1_fusionType-maxTempPool_num_epochs-23-aumented-data.pth'
    head, tail = os.path.split(classifier_file)

    # info = 'TemporalCongif:'+'videoBlockLength:'+str(videoBlockLength)+'-BlockOverlap:'+str(overlappingBlock)+'-videoSegmentLength:'+str(videoSegmentLength) +'-SegmentOverlap:'+str(overlappingSegment)+ '-numDynamImgsPerBlock:'+str(numDynamicImgsPerBlock)+'-MODEL:'+tail
    # print(info)
    # # offline(dataloaders_dict['test'], saliency_tester, typePersonDetector, h, w, plot)
    # classifier_file = 'ANOMALYCRIME/checkpoints/resnet18_Finetuned-False-_di-3_fusionType-maxTempPool_num_epochs-23_videoSegmentLength-30_positionSegment-begin-FINAL.pth'
    # classifier_file = 'ANOMALYCRIME/checkpoints/resnet18_Finetuned-False-_di-1_fusionType-maxTempPool_num_epochs-30-aumented-data.pth'
    # classifier_file = 'ANOMALYCRIME/checkpoints/Model-transfered-fine.pth'
    # classifier_file = 'ANOMALYCRIME/checkpoints/resnet18_Finetuned-True-_di-1_fusionType-maxTempPool_num_epochs-30-aumented-data-30.pth'
    # print('DEvice: ', device)
    config = {'videoBlockLength': videoBlockLength,
            'BlockOverlap':overlappingBlock,
            'videoSegmentLength': videoSegmentLength,
            'SegmentOverlap': overlappingSegment,
            'numDynamicImgsPerBlock': numDynamicImgsPerBlock,
            'model': tail}

    if str(device) == 'cpu':
        classifier = torch.load(classifier_file, map_location=torch.device('cpu'))
    else:
        classifier = torch.load(classifier_file)
        classifier = classifier.cuda()
    classifier = classifier.eval()
    classifier.inferenceMode(numDynamicImgsPerBlock)

    # print('Model device: ',next(classifier.parameters()).device)
    
    # tester = Tester(classifier, None, device, None, None)
    tester = Tester(model=classifier, dataloader=None, loss=None, device=device, numDiPerVideos=None, plot_samples=False)
    spatioTemporalDetection(anomalyDataset, tester, saliency_tester, typePersonDetector, h, w, plot, video_name, delay, device)
    # temporalTest(anomalyDataset, tail, config, tester, saliency_tester, typePersonDetector, h, w, plot, video_name, delay)
    
    # temporalTest(waqas_dataset, tester, saliency_tester, typePersonDetector, h, w, plot, video_name, delay, 'waqas')
       
__main__()


