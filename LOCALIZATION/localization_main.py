
import sys
# import include
# sys.path.insert(1,'/Users/davidchoqueluqueroman/Desktop/PAPERS-CODIGOS/violencedetection2')
sys.path.insert(1, '/media/david/datos/PAPERS-SOURCE_CODE/violencedetection')
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


def offline(dataloader, saliency_tester, type_person_detector, h, w, plot):
    person_model, classes = getPersonDetectorModel(type_person_detector)
    data_rows = []
    for i, data in enumerate(dataloader, 0):
        print("-" * 150)
        #di_images = [1,ndis,3,224,224]
        dis_images, labels, video_name, bbox_segments = data
        
        # - dis_images:  <class 'torch.Tensor'> torch.Size([1, 3, 224, 224])
        # - video_name:  <class 'tuple'>
        # - labels:  <class 'torch.Tensor'> torch.Size([1])
        # - bbox_segments:  <class 'list'> [[('frame645.jpg',), tensor([0]), tensor([72]), tensor([67]), tensor([133]), tensor([148])], ...,[]

        bbox_segments = np.array(bbox_segments)
        # dis_images = dis_images / 2 + 0.5 
        ttttt = dis_images.numpy()[0].transpose(1, 2, 0)
        plt.imshow(ttttt)
        plt.title('Input for Classifier')
        plt.show()
        #bbox_segments:  1 30 [('frame765.jpg',) tensor([0]) tensor([69]) tensor([80]) tensor([142]) tensor([152])]

        # print('bbox_segments: ', len(bbox_segments), len(bbox_segments[0]), bbox_segments[0][0])
        #video_raw_frames = np.array(video_raw_frames)
        #print('video_raw_frames: ', type(video_raw_frames[0][0]), len(video_raw_frames[0]), video_raw_frames[0][0].size())#[[[1, 240, 320, 3], [1, 240, 320, 3].[1, 240, 320, 3], ...]]
        ######################## dynamic images
        l_source_frames = []
        l_di_images = [] # to plot
        dis_images = dis_images.detach().cpu()  # torch.Size([1, 3, 224, 224])
        
        
        # dis_images = torch.squeeze(dis_images, 0) ## to num dynamic images > 1 and minibatch == 1
        for di_image in dis_images:
            # di_image = di_image / 2 + 0.5 
            # di_image = resize2raw_transform(di_image)
            di_image = di_image.numpy().transpose(1, 2, 0)
            l_di_images.append(di_image)
            # print('di_image: ', di_image.shape)

        ######################## mask
        
        masks = saliency_tester.compute_mask(dis_images, labels)  #[1, 1, 224, 224]
        
        mascara = masks.detach().cpu()#(1, 1, 224, 224)
        mascara = torch.squeeze(mascara,0)#(1, 224, 224)
        mascara = saliency_tester.min_max_normalize_tensor(mascara) #to 0-1
        mascara = mascara.numpy().transpose(1, 2, 0)  
        mascara = np.concatenate((mascara,) * 3, axis=-1)

        ##resize 
        mascara = cv2.resize(mascara, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
        
        plt.imshow(mascara)
        plt.title('MASCARA')
        plt.show()

        mascara_dn = np.uint8(255*mascara)
        mascara_dn = cv2.fastNlMeansDenoisingColored(mascara_dn, None, 10, 10, 7, 21)
        # print('mascara :', mascara.shape)

        plt.imshow(mascara_dn)
        plt.title('MASCARA denoised')
        plt.show()

        
        masks = torch.from_numpy(mascara_dn).float()  #torch.Size([224, 224, 3])
        masks = masks.permute(2, 0, 1)  
        masks = torch.unsqueeze(masks,dim=0) #masks : torch.Size([1, 3, 240, 320]
        
        # unsqueeze(0)
        # masks = torch.squeeze(masks, 0)  #tensor [ndis,1,224,224]
        # masks = masks.detach().cpu()
        masks = saliency_tester.min_max_normalize_tensor(masks) #to 0-1
        source_frames = masks

        video_prepoc_saliencies = []
        # video_prepoc_saliencies2 = []
        # l_imgs_processing = []

        #print('raw frames: ', video_raw_frames.size()) #torch.Size([1, 1, 30, 3, 240, 320])
        for idx_dyn_img, source_frame in enumerate(source_frames):  #get score from dyanmic regions
            # source_frame = resize2raw_transform(source_frame)
            #source_frame = tensor2numpy(source_frame)
            
            source_frame = source_frame.numpy().transpose(1,2,0)
            print('source frane: ', source_frame.shape)
            saliency_bboxes, preprocesing_reults = localization_utils.computeBoundingBoxFromMask(source_frame)  #only one by segment
            saliency_bboxes = localization_utils.removeInsideSmallAreas(saliency_bboxes)
            # saliency_bboxes.append(localization_utils.randomBBox(240, 320))
            # saliency_bboxes.append(localization_utils.randomBBox(240, 320))
            # saliency_bboxes.append(localization_utils.randomBBox(240,320))
            # scoring.getScoresFromRegions(video_name[0], bbox_segments, saliency_bboxes,classifier, transforms_dataset['test'])

            # print('saliency_bboxes: ', type(saliency_bboxes),len(saliency_bboxes),saliency_bboxes[0])
            video_prepoc_saliencies.append({
                'saliency bboxes': saliency_bboxes,
                'preprocesing': preprocesing_reults
            })            
        
        segment_info = []
        for bbox_segment in bbox_segments:
            # print('bbox_segment: ',bbox_segment)
            #read frames of segment
            frames_names, real_frames, real_bboxes = localization_utils.getFramesFromSegment(video_name[0], bbox_segment, 'all')
            segment_info.append({
                'frames_names':frames_names,
                'real_frames':real_frames,
                'real_bboxes': real_bboxes})
          
            # img_test = tt(real_frames[0])
            # img_test = torch.unsqueeze(img_test,dim=0)
    
        for index, segment_real_info in enumerate(segment_info): #di by di
            #person detection
            persons_in_segment = []
            anomalous_regions = []  # to plot
            persons_in_frame = []
            persons_segment_filtered = []  #only to vizualice
            print('Real frames shot: ', len(segment_real_info['real_frames']))
            for idx, frame in enumerate(segment_real_info['real_frames']):
                if type_person_detector == constants.YOLO:
                    img_size = 416
                    conf_thres = 0.8
                    nms_thres = 0.4
                    persons_in_frame = localization_utils.personDetectionInFrameYolo(person_model, img_size, conf_thres, nms_thres, classes, frame)
                    # print('persons_in_frame MaskRCNN: ', len(persons_in_frame))
                elif type_person_detector == constants.MASKRCNN:
                    mask_rcnn_threshold = 0.4
                    persons_in_frame = MaskRCNN.personDetectionInFrameMaskRCNN(person_model, frame, mask_rcnn_threshold)
                # print('num Persons in frame: ', len(persons_in_frame))
                persons_in_segment.append(persons_in_frame)
        
            #refinement in segment
            for i,persons_in_frame in enumerate(persons_in_segment):
                persons_filtered, only_joined_regions = localization_utils.filterClosePersonsInFrame(persons_in_frame, 20)
                persons_segment_filtered.append(only_joined_regions)
                # print('persons_in_frame filter: ', len(persons_filtered))
                anomalous_regions_in_frame = localization_utils.findAnomalyRegionsOnFrame(persons_filtered, video_prepoc_saliencies[index]['saliency bboxes'], 0.3)
                # anomalous_regions_in_frame = sorted(anomalous_regions_in_frame, key=lambda x: x.iou, reverse=True)
                
                anomalous_regions.extend(anomalous_regions_in_frame)
                # print('--------------- anomalous_regions: ', len(anomalous_regions))
                anomalous_regions.sort(key=lambda x: x.iou, reverse=True) #sort by iuo to get only one anomalous regions
            ######################################              
            segmentBox = localization_utils.getSegmentBBox(segment_real_info['real_bboxes'])
            # l = [segmentBox]
            # print('=========== Ground Truth Score =============')
            # scoring.getScoresFromRegions(video_name[0], bbox_segments, l, classifier, transforms_dataset['test'])
           
            # for b in segment_real_info['real_bboxes']:
            #     b.score = l[0].score
            
            ######################################
            if len(anomalous_regions) == 0:
                # print('No anomalous regions found...')
                iou = localization_utils.IOU(segmentBox,None)
            else:
                # segmentBox = localization_utils.getSegmentBBox(segment_real_info['real_bboxes'])
                iou = localization_utils.IOU(segmentBox,anomalous_regions[0])
            row = [frames_names[0], iou]
            data_rows.append(row)
        
            if plot:
                font_size = 9
                subplot_r = 2
                subplot_c = 5
                img_source = source_frames[0].numpy().transpose(1, 2, 0)
                # img_source = l_source_frames[index]

                # di_image = dis_images.numpy()[0].transpose(1, 2, 0)
                di_image = ttttt
                # di_image = di_image / 2 + 0.5

                # preprocesing_reults = video_prepoc_saliencies[index]['preprocesing']
                preprocesing_reults = np.empty( (subplot_r,subplot_c), dtype=tuple)
                preprocesing_reults[0,0] = (img_source,'image source')
                preprocesing_reults[0,1]=(video_prepoc_saliencies[index]['preprocesing'][0], 'thresholding')
                preprocesing_reults[0,2]=(video_prepoc_saliencies[index]['preprocesing'][1], 'morpho')
                preprocesing_reults[0,3]=(video_prepoc_saliencies[index]['preprocesing'][2], 'contours')
                preprocesing_reults[0,4]=(video_prepoc_saliencies[index]['preprocesing'][3], 'bboxes')

                real_frames = segment_info[index]['real_frames']
                real_bboxes = segment_info[index]['real_bboxes']
                
                saliency_bboxes = video_prepoc_saliencies[index]['saliency bboxes']
                myplot(subplot_r, subplot_c, font_size, preprocesing_reults, di_image, real_frames, real_bboxes, saliency_bboxes, persons_in_segment, persons_segment_filtered, anomalous_regions)
                
    
    # ############# MAP #################
    ns_fpr, ns_tpr, _ = roc_curve(y_true, y_)
    ns_auc = roc_auc_score(testy, ns_probs)
    # lr_auc = roc_auc_score(testy, lr_probs)
    # print('data rows: ', len(data_rows))
    # df = pd.DataFrame(data_rows, columns=['path', 'iou'])
    # df['tp/fp'] = df['iou'].apply(lambda x: 'TP' if x >= 0.5 else 'FP')
    # localization_utils.mAP(df)

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

def temporalTest(anomalyDataset, class_tester, saliency_tester, type_person_detector, h, w, plot, only_video_name, delay, datasetType='UCF2Local'):
    person_model, classes = getPersonDetectorModel(type_person_detector)
    bbox_last = None
    distance_th = 35.5
    thres_intersec_lastbbox_current = 0.4
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
    y_pred_scored_based = []
    video_aucs = []
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
            print('dskfgljksdhgkjshgksglksdjglksdjglksdgiuegjhfgiowijgowo')

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
        block_dinamyc_images, idx_next_block, block_boxes_info = anomalyDataset.computeBlockDynamicImg(idx_video, idx_next_block=0, skip=skip)
        print('block_dinamyc_images: ', block_dinamyc_images.size())
        # dis_images, segment_info, idx_next_segment = anomalyDataset.computeSegmentDynamicImg(idx_video=idx_video, idx_next_segment=0)
        num_block = 0
        video_y_pred = []
        video_y_gt = []
        numFrames = anomalyDataset.getNumFrames(idx_video)
        while (block_dinamyc_images is not None): #dis_images : torch.Size([3, 224, 224])
            # print('Dynamic image block: ', block_dinamyc_images.size()) #torch.Size([1, 3, 224, 224])
            # dynamic_img = torch.unsqueeze(dis_images, dim=0)
            # print('---numBlock: ', num_block, '--idx_next_block:', idx_next_block)
            prediction, score = class_tester.predict(block_dinamyc_images)
            tp, fp, y_block_pred, y_score_based = localization_utils.countTruePositiveFalsePositive(block_boxes_info, prediction, score,0.8)
            video_y_pred.extend(y_block_pred)
            y_pred_scored_based.extend(y_score_based)
            if datasetType == 'waqas':
                p, n, y_block_truth = localization_utils.countTemporalGroundTruth(block_boxes_info)
            else:
                p, n, y_block_truth = localization_utils.countPositiveFramesNegativeFrames(block_boxes_info)
            video_y_gt.extend(y_block_truth)
            num_pos_frame += p
            num_neg_frame += n
            total_fp += fp
            total_tp += tp
            y_truth.extend(y_block_truth)
            y_pred.extend(y_block_pred)
            num_block += 1
            if plot:
                # if datasetType == 'waqas':
                # else:
                frames_names, real_frames, real_bboxes = localization_utils.getFramesFromBlock(video_name[0], block_boxes_info)

                # block_dinamyc_images = torch.unsqueeze(block_dinamyc_images, 0)
                dynamic_image_plot = saliency_tester.min_max_normalize_tensor(block_dinamyc_images) 
                dynamic_image_plot = dynamic_image_plot.numpy()[0].transpose(1, 2, 0)
                dynamic_image_plot = cv2.resize(dynamic_image_plot, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(dynamic_image_plot,str(score),(0,25), font, 0.6,constants.yellow,2,cv2.LINE_AA)
                # plt.imshow(dynamic_image_plot)
                # plt.show()
                

                pos_x = 20
                sep = 400
                for i, frame in enumerate(real_frames):
                    frame = np.array(frame)
                    # font = cv2.FONT_HERSHEY_SIMPLEX
                    # cv2.putText(frame,str(score),(0,25), font, 0.6,constants.red,2,cv2.LINE_AA)
                    wn = frame.shape[0]
                    hn = frame.shape[1]
                    if prediction == 1:
                    # if score >= 0.8:
                        cv2.rectangle(frame, (0, 0), (hn, wn), constants.magenta, 4)

                    gt_bbox = real_bboxes[i]
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
            
            block_dinamyc_images, idx_next_block, block_boxes_info = anomalyDataset.computeBlockDynamicImg(idx_video, idx_next_block=idx_next_block,skip=skip)
        # vid_auc = roc_auc_score(video_y_gt, video_y_pred)
        # print('Video: ', video_name[0],str(vid_auc))
            
        # return segment_inf
    # fpr, tpr, _ = roc_curve(y_truth, y_pred_scored_based)
    # fpr, tpr, _ = roc_curve(y_truth, y_pred)
    # lr_auc = roc_auc_score(y_truth, y_pred)
    # vauc = auc(fpr,tpr)
    # plt.plot(fpr, tpr, marker='.', label='test')
        # axis labels
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')

    num_positive_pred = len([i for i in y_pred if i >= 0.5])
    false_alarm_rate = num_positive_pred / len(y_pred)
    print('False alarm rate: ', str(false_alarm_rate))
    print('num positive: ', str(num_positive_pred))
    # print(y_truth)

        # show the legend
        # pyplot.legend()
        # show the plot
    # plt.show()
    print('Frames processed: ', len(y_truth), len(y_pred))
    # print('AUC: ', str(lr_auc), vauc)

def online(anomalyDataset, class_tester, saliency_tester, type_person_detector, h, w, plot, only_video_name, delay, device):
    person_model, classes = getPersonDetectorModel(type_person_detector, device)
    # person_model = person_model.to(device)
    # print('model cuda: ', next(person_model.parameters()).is_cuda)
    bbox_last = None
    distance_th = 35.5
    thres_intersec_lastbbox_current = 0.4
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
    num_iter = 1
    fpsMeter = FPSMeter()

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
        # di_spend_times = []
        bbox_last = None
        print("-" * 150, 'video No: ', idx_video)
        video_name, label = data
        video_name = [video_name]
        label = torch.tensor(label)

        #################### BLOCK ####################
        # block_dinamyc_images, idx_next_block, block_boxes_info = anomalyDataset.computeBlockDynamicImg(idx_video, idx_next_block=0, skip=1)
       
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
        dis_images, segment_info, idx_next_segment, spend_time_dyImg = anomalyDataset.computeSegmentDynamicImg(idx_video=idx_video, idx_next_segment=0)
        # fpsMeter.update(spend_time_dyImg)
        inference_start_time = time.time()
        prediction, score = class_tester.predict(torch.unsqueeze(dis_images, dim=0), num_iter)
        # torch.cuda.synchronize()
        inference_end_time = time.time()
        # inference_time = inference_end_time - inference_start_time
        # total_time = spend_time_dyImg + inference_time
        # print('Time in seconds: ', total_time)
        # fpsMeter.update(inference_time)
        num_segment = 0
        while (dis_images is not None): #dis_images : torch.Size([3, 224, 224])
            num_segment += 1            
            print(video_name, '---segment No: ', str(num_segment),'-----', dis_images.size(), len(segment_info), dis_images.is_cuda)
            
            # print('- dis_images: ', type(dis_images), dis_images.size())
            # print('- video_name: ', type(video_name), video_name)
            # print('- label: ', type(label))
            # print('- segment_info: ', type(segment_info), segment_info[0])
            # - dis_images:  <class 'torch.Tensor'>
            # - video_name:  <class 'str'>
            # - labels:  <class 'int'>
            # - bbox_segments:  <class 'list'>  [['frame645.jpg', 0, 72, 67, 133, 148],..., []
            dis_images = torch.unsqueeze(dis_images, 0)
            segment_info = default_collate([segment_info])
            segment_info = np.array(segment_info)
            dis_images = dis_images / 2 + 0.5 
            ttttt = saliency_tester.min_max_normalize_tensor(dis_images) 
            ttttt = ttttt.numpy()[0].transpose(1, 2, 0)
            ttttt = cv2.resize(ttttt, dsize=(w, h), interpolation=cv2.INTER_CUBIC)

            # plt.imshow(ttttt)
            # plt.title('Input for Classifier')
            # plt.show()
            # cv2.imshow('image', ttttt)
            # k = cv2.waitKey(0)
            # if k == 27:         # wait for ESC key to exit
            #     cv2.destroyAllWindows()

            ######################## mask
            mask_start_time = time.time()
            masks = saliency_tester.compute_mask(dis_images, label)  #[1, 1, 224, 224]
            torch.cuda.synchronize()
            mask_end_time = time.time()
            mask_time = mask_end_time - mask_start_time

            fpsMeter.update(mask_time)
            
            mascara = masks.detach().cpu()#(1, 1, 224, 224)
            mascara = torch.squeeze(mascara,0)#(1, 224, 224)
            mascara = saliency_tester.min_max_normalize_tensor(mascara)  #to 0-1
            mascara = mascara.numpy().transpose(1, 2, 0)

            ##resize 
            mascara = cv2.resize(mascara, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
            
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
            track_start_time = time.time()         
            if bbox_last is not None:
                bboxes_with_high_core = []
                for bbox in saliency_bboxes:
                    pred_distance = localization_utils.distance(bbox.center, bbox_last.center)
                    if pred_distance <= distance_th or bbox.percentajeArea(localization_utils.intersetionArea(bbox,bbox_last)) >= thres_intersec_lastbbox_current:
                        bbox.score = pred_distance
                        bboxes_with_high_core.append(bbox)
                if len(bboxes_with_high_core) > 0:
                    bboxes_with_high_core.sort(key=lambda x: x.score)
                    saliency_bboxes = bboxes_with_high_core
                else:
                    # print('FFFFail in close lastBBox and saliency bboxes...')
                    saliency_bboxes = [bbox_last]

            if len(saliency_bboxes) == 0:
                saliency_bboxes = [BoundingBox(Point(0,0),Point(w,h))]
            
            track_end_time = time.time()
            track_time = track_end_time - track_start_time
            
            frames_names, real_frames, real_bboxes = localization_utils.getFramesFromSegment(video_name[0], segment_info, 'all')
            persons_in_segment = []
            persons_segment_filtered = []
            anomalous_regions = []  # to plot
            segmentBox = localization_utils.getSegmentBBox(real_bboxes)
            
            # print('Real frames shot: ', len(real_frames))
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
                    detector_start_time = time.time()
                    persons_in_frame = localization_utils.personDetectionInFrameYolo(person_model, img_size, conf_thres,
                                                                                    nms_thres, classes, frame, device)
                    torch.cuda.synchronize()
                    detector_end_time = time.time()
                    detector_time = detector_end_time-detector_start_time

                elif type_person_detector == constants.MASKRCNN:
                    mask_rcnn_threshold = 0.4
                    persons_in_frame = MaskRCNN.personDetectionInFrameMaskRCNN(person_model, frame, mask_rcnn_threshold)
                # print('--num Persons in frame: ', len(persons_in_frame))
                refinement_start_time = time.time()
                if len(persons_in_frame) > 0:
                    persons_in_segment.append(persons_in_frame)
                    iou_threshold = 0.3
                    thres_intersec_person_saliency = 0.5
                    thresh_close_persons = 20
                
                    if bbox_last is not None:
                        print('='*10, ' ONLINE ')
                        dynamic_region = saliency_bboxes[0]
                        persons_filtered, only_joined_regions = localization_utils.filterClosePersonsInFrame(persons_in_frame, thresh_close_persons)
                        # print('---Persons after filter close: ', len(persons_filtered), len(only_joined_regions))
                        # persons_segment_filtered.append(persons_filtered)
                        for person in persons_filtered:
                            iou = localization_utils.IOU(person, dynamic_region)
                            # print('----IOU (person and dynamic region): ', str(iou))
                            if iou >= iou_threshold:
                                abnorm_bbox = localization_utils.joinBBoxes(person,dynamic_region)
                                abnorm_bbox.score = iou
                                anomalous_regions.append(abnorm_bbox)
                        
                        if len(anomalous_regions) > 0:
                            anomalous_regions.sort(key=lambda x: x.iou, reverse=True) #sort by iuo to get only one anomalous regions
                            anomalous_regions[0].score = -1
                            # bbox_last = anomalous_regions[0]
                            break

                    else:
                        print('='*10, ' FIRST SHOT ')
                        persons_filtered, only_joined_regions = localization_utils.filterClosePersonsInFrame(persons_in_frame, thresh_close_persons)
                        # print('---Persons after filter close: ', len(persons_filtered), len(only_joined_regions))
                        # persons_segment_filtered.append(persons_filtered)
                        
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
                            
                        if len(anomalous_regions) > 0:
                            anomalous_regions.sort(key=lambda x: x.iou, reverse=True) #sort by iuo to get only one anomalous regions
                            anomalous_regions[0].score = -1
                            bbox_last = anomalous_regions[0]
                            break
                        else: # 
                            temporal_ious_regions.sort(key=lambda x: x.iou, reverse=True)
                            anomalous_regions.append(temporal_ious_regions[0])
                            break
                    if len(anomalous_regions) > 0 :
                        break
                    
                else:
                    persons_in_segment.append(None) #only to plot
                refinement_end_time = time.time()
                refinement_time = refinement_end_time - refinement_start_time
                # fpsMeter.update(refinement_time+detector_time)
            if len(anomalous_regions) == 0:
                # print('Tracking Algorithm FAIL!!!!. Using last localization...')
                anomalous_regions.append(bbox_last)   
            # fpsMeter.update(mask_time+preproc_time+track_time)
            
            # # iou = localization_utils.IOU(segmentBox,anomalous_regions[0])
            # # row = [video_name[0]+'---segment No: '+str(num_segment), iou]
            
            # # data_rows_video.append(row)
            # #################### BLOCK ####################
            # # if idx_next_segment > idx_next_block:
            # #     block_dinamyc_images, idx_next_block, block_boxes_info = anomalyDataset.computeBlockDynamicImg(idx_video, idx_next_block=idx_next_block, skip=1)
            # #     # dynamic_img = torch.unsqueeze(dis_images, dim=0)
            # #     # print('block_dinamyc_images: ', block_dinamyc_images.size())
            # #     prediction, score = class_tester.predict(block_dinamyc_images)
            # #     tp, fp, y_block_pre, y_score_based = localization_utils.countTruePositiveFalsePositive(block_boxes_info, prediction, score, threshold=0)
            # #     p, n, y_block_truth = localization_utils.countPositiveFramesNegativeFrames(block_boxes_info)
            # #     num_pos_frame += p
            # #     num_neg_frame += n
            # #     total_fp += fp
            # #     total_tp += tp
            # #     y_truth.extend(y_block_truth)
            # #     y_pred.extend(y_block_pred)
            dis_images, segment_info, idx_next_segment, spend_time_dyImg = anomalyDataset.computeSegmentDynamicImg(idx_video=idx_video, idx_next_segment=idx_next_segment)
            # fpsMeter.update(spend_time_dyImg)
            if dis_images is not None:
                inference_start_time = time.time()
                prediction, score = class_tester.predict(torch.unsqueeze(dis_images, dim=0), num_iter)
                torch.cuda.synchronize()
                inference_end_time = time.time()
                inference_time = inference_end_time - inference_start_time
                # fpsMeter.update(inference_time)
            
            # if dis_images is not None:
            #     print('dis_images: ', type(dis_images))
            #     start_time = time.time()
            #     prediction, score = class_tester.predict(torch.unsqueeze(dis_images, dim=0), num_iter)
            #     torch.cuda.synchronize()
            #     end_time = time.time()
            #     inference_time = end_time-start_time
            #     fpsMeter.update(spend_time_dyImg + inference_time)
                
            # # bbox_last = anomalous_regions[0]        
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
            
        # average_time = np.mean(di_spend_times)
        # print('Average time: ', str(average_time), len(di_spend_times)) 
        # print('Frames per Second: ', str(1/average_time))      
        # with open("videos_scores.txt", "w") as txt_file:
        #     for line in videos_scores:
        #         txt_file.write(" ".join(line) + "\n")

        # print(videos_scores)

        # fpr, tpr, _ = roc_curve(y_truth, y_pred)
        # lr_auc = roc_auc_score(y_truth, y_pred)
        # plt.plot(fpr, tpr, marker='.', label='test')
        # # axis labels
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # # show the legend
        # # pyplot.legend()
        # # show the plot
        # plt.show()
        # print('AUC: ', str(lr_auc))
        ############# MAP #################
        # print('data rows: ', len(data_rows))
        # df = pd.DataFrame(data_rows, columns=['path', 'iou'])
        # total_rows = df.shape[0]
        # seriesObj = df.apply(lambda x: True if x['iou'] <0.5 else False , axis=1)
        # numOfRows = len(seriesObj[seriesObj == True].index)
        # localization_error = numOfRows/total_rows
        # print('BAd localizations num: ', numOfRows, ', Total frames: ', total_rows, 'Loc Error: ', str(localization_error))
        # df = df.sort_values('iou',ascending=False)
        # df = df.reset_index(drop=True)
        # export_csv = df.to_csv ('metrics_yolo.csv', index = None, header=True)
        # df['tp/fp'] = df['iou'].apply(lambda x: 'TP' if x >= 0.5 else 'FP')   


        # for frame in segment:
        #     # print('frame: ', frame)
        #     num_frame = int(frame[len(frame) - 7:-4])
        #     if num_frame != int(data[num_frame, 5]):
        #         sys.exit('Houston we have a problem: index frame does not equal to the bbox file!!!')
            
        #     flac = int(data[num_frame,6]) # 1 if is occluded: no plot the bbox
        #     xmin = int(data[num_frame, 1])
        #     ymin= int(data[num_frame, 2])
        #     xmax = int(data[num_frame, 3])
        #     ymax = int(data[num_frame, 4])
        #     # print(type(frame), type(flac), type(xmin), type(ymin))
        #     info_frame = [frame, flac, xmin, ymin, xmax, ymax]
        #     segment_info.append(info_frame)
        # return segment_inf
        row = [video_name[0], round(fpsMeter.fps(),2), round(fpsMeter.mspf(),2)]
        # print(row)
        data_rows.append(row)

    # data_rows = zip(data_rows)
    df = pd.DataFrame(data_rows, columns=["video", "fps", "mspf"])
    df.to_csv('fpsAnomaly.csv', index=False)
    # fpsMeter.print_statistics()
    ################### ROC - AUC ############################
    # fpr, tpr, _ = roc_curve(y_truth, y_pred)
    # lr_auc = roc_auc_score(y_truth, y_pred)
    # plt.plot(fpr, tpr, marker='.', label='test')

    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.show()
    # print('AUC: ', str(lr_auc))

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
    path_dataset = constants.PATH_UCFCRIME2LOCAL_FRAMES_REDUCED
    train_videos_path = os.path.join(constants.PATH_UCFCRIME2LOCAL_README, 'Train_split_AD.txt')
    test_videos_path = os.path.join(constants.PATH_UCFCRIME2LOCAL_README, 'Test_split_AD.txt')
# path_dataset, test_videos_path, batch_size, num_workers, videoBlockLength,
#                     numDynamicImgsPerBlock, transform, videoSegmentLength, shuffle,overlappingBlock, overlappingSegment
    # videoBlockLength = 30
    only_anomalous = True
    dataloader_test, test_names, test_labels, anomalyDataset = anomalyInitializeDataset.initialize_test_anomaly_dataset(path_dataset,
                                                        test_videos_path, batch_size, num_workers, videoBlockLength, numDynamicImgsPerBlock,
                                                        transforms_dataset['test'],
                                                        videoSegmentLength, shuffle, overlappingBlock, overlappingSegment, only_anomalous)
    
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
    # # offline(dataloaders_dict['test'], saliency_tester, typePersonDetector, h, w, plot)
    classifier_file = 'ANOMALYCRIME/checkpoints/resnet18_Finetuned-False-_di-3_fusionType-maxTempPool_num_epochs-23_videoSegmentLength-30_positionSegment-begin-FINAL.pth'
    # classifier_file = 'ANOMALYCRIME/checkpoints/resnet18_Finetuned-False-_di-1_fusionType-maxTempPool_num_epochs-30-aumented-data.pth'
    # classifier_file = 'ANOMALYCRIME/checkpoints/Model-transfered-fine.pth'
    # classifier_file = 'ANOMALYCRIME/checkpoints/resnet18_Finetuned-True-_di-1_fusionType-maxTempPool_num_epochs-30-aumented-data-30.pth'
    if str(device) == 'cpu':
        classifier = torch.load(classifier_file, map_location=torch.device('cpu'))
    else:
        classifier = torch.load(classifier_file)
        classifier = classifier.cuda()
    classifier = classifier.eval()
    classifier.inferenceMode(numDynamicImgsPerBlock)
    
    tester = Tester(classifier, None, device, None, None)
    online(anomalyDataset, tester, saliency_tester, typePersonDetector, h, w, plot, video_name, delay, device)
    # temporalTest(anomalyDataset, tester, saliency_tester, typePersonDetector, h, w, plot, video_name, delay)
    # temporalTest(waqas_dataset, tester, saliency_tester, typePersonDetector, h, w, plot, video_name, delay, 'waqas')
    
   
__main__()


