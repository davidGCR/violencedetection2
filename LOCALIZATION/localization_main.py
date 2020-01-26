
import sys
# import include
sys.path.insert(1,'/Users/davidchoqueluqueroman/Desktop/PAPERS-CODIGOS/violencedetection2')
# sys.path.insert(1, '/media/david/datos/PAPERS-SOURCE_CODE/violencedetection')
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

def getPersonDetectorModel(detector_type):
    classes = None
    if detector_type == constants.YOLO:
        img_size = 416
        weights_path = "YOLOv3/weights/yolov3.weights"
        class_path = "YOLOv3/data/coco.names"
        model_def = "YOLOv3/config/yolov3.cfg"
        person_model, classes = yolo_inference.initializeYoloV3(img_size, class_path, model_def, weights_path)
        
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
    print('data rows: ', len(data_rows))
    df = pd.DataFrame(data_rows, columns=['path', 'iou'])
    df['tp/fp'] = df['iou'].apply(lambda x: 'TP' if x >= 0.5 else 'FP')
    localization_utils.mAP(df)

def plotOpencv(images, gt_bboxes, persons_in_segment, bbox_predictions, dynamic_image, mascara, preprocess, saliency_bboxes):
    red = (0, 0, 255)
    green = (0, 255, 0)
    blue = (255, 0, 0)
    yellow = (0,255,255)
    for idx,image in enumerate(images):
        
        image = np.array(image)
        # print('eeeeeeeeeeeeeeeehhhhhhhhhhhhhhhhhhhh ',image.shape, type(dynamic_image), dynamic_image.shape)
        for gt_bbox in gt_bboxes:
            cv2.rectangle(image, (int(gt_bbox.pmin.x), int(gt_bbox.pmin.y)), (int(gt_bbox.pmax.x), int(gt_bbox.pmax.y)), red, 1)
            # cv2.rectangle(image, (5, 5), (44, 45), color=(0, 255, 0), thickness=2)
        for pred in bbox_predictions:
            if pred is not None:
                cv2.rectangle(image, (int(pred.pmin.x), int(pred.pmin.y)), (int(pred.pmax.x), int(pred.pmax.y)), blue, 2)

        if idx < len(persons_in_segment):
            if persons_in_segment[idx] is not  None:
                for person in persons_in_segment[idx]:
                    cv2.rectangle(image, (int(person.pmin.x), int(person.pmin.y)), (int(person.pmax.x), int(person.pmax.y)), green, 1)

        for s_bbox in saliency_bboxes:
            cv2.rectangle(image, (int(s_bbox.pmin.x), int(s_bbox.pmin.y)), (int(s_bbox.pmax.x), int(s_bbox.pmax.y)), yellow, 1)
        
        # aa = np.concatenate((image, image), axis=1)
        cv2.imshow('image', image)
        cv2.imshow('dynamic', dynamic_image)
        cv2.imshow('mascara', mascara)
        cv2.imshow('preprocess', preprocess)

        cv2.namedWindow("image");#x,y
        cv2.moveWindow("image", 20, 100);

        cv2.namedWindow("dynamic");
        cv2.moveWindow("dynamic", 400, 100);
        
        cv2.namedWindow("mascara");
        cv2.moveWindow("mascara", 800, 100);

        cv2.namedWindow("preprocess");
        cv2.moveWindow("preprocess", 20,500);
        # k = cv2.waitKey(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break   
        # if k == 27:         # wait for ESC key to exit
        #     cv2.destroyAllWindows()

def online(anomalyDataset, saliency_tester, type_person_detector, h, w, plot, video_name):
    person_model, classes = getPersonDetectorModel(type_person_detector)
    bbox_last = None
    distance_th = 35.5
    thres_intersec_lastbbox_current = 0.4
    data_rows = []
    indx_flac = -1
    video_dynamic_images = []
    video_mascaras = []
    video_preprocesing_outs = []
    video_saliency_bboxes = []
    video_real_frames = []
    video_persons_in_segment = []
    video_persons_segment_filtered = []
    video_anomalous_regions = []
    video_real_bboxes = []
    num_video_segments = 0
    videos_scores = []

    for idx_video, data in enumerate(anomalyDataset):
        indx_flac = idx_video
        if video_name is not None and indx_flac==0:
            if indx_flac==0:
                idx = anomalyDataset.getindex(video_name)
                if idx is not None:
                    data = anomalyDataset[idx]
                    idx_video = idx
                    print('TEsting only one video...' , video_name)
                else:
                    print('No valid video...')
            else:
                break

        if video_name is not None and indx_flac>0:
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
        dis_images, segment_info, idx_next_segment = anomalyDataset.computeSegmentDynamicImg(idx_video=idx_video, idx_next_segment=0)
        num_segment = 0
        while (dis_images is not None):
            num_segment += 1
            # num_video_segments += 1
            # - dis_images:  <class 'torch.Tensor'> torch.Size([1, 3, 224, 224])
            # - video_name:  <class 'tuple'>
            # - labels:  <class 'torch.Tensor'> torch.Size([1])
            # - bbox_segments:  <class 'list'> [[('frame645.jpg',), tensor([0]), tensor([72]), tensor([67]), tensor([133]), tensor([148])], ...,[]
            print(video_name, '---segment No: ', str(num_segment),'-----', dis_images.size(), len(segment_info))
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
            # dis_images = dis_images / 2 + 0.5 
            ttttt = saliency_tester.min_max_normalize_tensor(dis_images) 
            ttttt = ttttt.numpy()[0].transpose(1, 2, 0)
            ttttt = cv2.resize(ttttt, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
            video_dynamic_images.append(ttttt)

            # plt.imshow(ttttt)
            # plt.title('Input for Classifier')
            # plt.show()
            # cv2.imshow('image', ttttt)
            # k = cv2.waitKey(0)
            # if k == 27:         # wait for ESC key to exit
            #     cv2.destroyAllWindows()
            

            ######################## mask
            masks = saliency_tester.compute_mask(dis_images, label)  #[1, 1, 224, 224]
            mascara = masks.detach().cpu()#(1, 1, 224, 224)
            mascara = torch.squeeze(mascara,0)#(1, 224, 224)
            mascara = saliency_tester.min_max_normalize_tensor(mascara)  #to 0-1
            mascara = mascara.numpy().transpose(1, 2, 0)
            ##resize 
            mascara = cv2.resize(mascara, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
            video_mascaras.append(mascara)
            
            # print('Mascara 0000(min: ',str(np.amin(mascara)),' - max: ', str(np.amax(mascara)))
            # print('mascara antest de todo: ', mascara.shape,' w: ',w,'h: ',h) 
            # plt.imshow(mascara)
            # plt.title('MASCARA')
            # plt.show()

            # mascara = np.uint8(255*mascara)
            # mascara = cv2.fastNlMeansDenoisingColored(mascara, None, 10, 10, 7, 21)
            # _min = np.amin(mascara)
            # _max = np.amax(mascara)
            # # print("min:", _min.item(), ", max:", _max.item())
            # mascara = (mascara - _min) / (_max - _min)

            # plt.imshow(mascara)
            # plt.title('MASCARA denoised')
            # plt.show()
            
            # mascara_by_summarized = np.multiply(np.stack((mascara,) * 3, axis=-1), ttttt)
            # mascara_gray = cv2.cvtColor(mascara, cv2.COLOR_BGR2GRAY)
           
            # mascara = mascara_gray
            # mascara_gray = 255*mascara_gray #between 0-255
            # mascara_gray = mascara_gray.astype('uint8')
            # ret, mascara_gray = cv2.threshold(mascara_gray, 5, 255, 0)

            # img = np.zeros((mascara_gray.shape[0], mascara_gray.shape[1], 3), dtype=np.uint8)
            # contours, hierarchy = cv2.findContours(mascara_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # for i in range(len(contours)):
            #     cv2.drawContours(img, contours, i, (0, 255, 0), 2, cv2.LINE_8, hierarchy, 0)
            
            # plt.imshow(mascara_gray)
            # plt.title('MASCARA gray')
            # plt.show()

            # plt.imshow(img)
            # plt.title('MASCARA gray contours')
            # plt.show()
            # mascara = mascara_by_summarized[:,:,0]

            
            # mascara = ttttt[:,:,0]
            saliency_bboxes, preprocesing_outs = localization_utils.computeBoundingBoxFromMask(mascara)  #only one by segment
            video_preprocesing_outs.append(preprocesing_outs)
            video_saliency_bboxes.append(saliency_bboxes)
            # min_px = np.amin(mascara)
            # max_px = np.amax(mascara)
            # #0:  (min:  0.21259843  - max:  0.9015748
            # print('Mascara (min: ',str(min_px),' - max: ', str(max_px))

            # for bbox in saliency_bboxes:
            #     crop = mascara[bbox.pmin.y:bbox.pmax.y, bbox.pmin.x:bbox.pmax.x]
            #     min_px = np.amin(crop)
            #     max_px = np.amax(crop)
            #     #0:  (min:  0.21259843  - max:  0.9015748
            #     print('Mascara Crop numpy (min: ',str(min_px),' - max: ', str(max_px))
            #     plt.imshow(crop)
            #     plt.show()
            # print('preprocesing_outs: ', len(preprocesing_outs))
            #metricas (no decir es mejor... 2 puntos porcentuales del valor-outperform) en esa base de datos- grafico explicar ejes...porque de los resultados
            #presentar errores
            saliency_bboxes = localization_utils.removeInsideSmallAreas(saliency_bboxes)            
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
                    # saliency_bboxes[0].score = -1
                else:
                    print('FFFFail in close lastBBox and saliency bboxes...')
                    saliency_bboxes = [bbox_last]

            if len(saliency_bboxes) == 0:
                saliency_bboxes = [BoundingBox(Point(0,0),Point(w,h))]
            
            frames_names, real_frames, real_bboxes = localization_utils.getFramesFromSegment(video_name[0], segment_info, 'all')
            video_real_frames.append(real_frames)
            video_real_bboxes.append(real_bboxes)
            
            persons_in_segment = []
            persons_segment_filtered = []
            anomalous_regions = []  # to plot
            segmentBox = localization_utils.getSegmentBBox(real_bboxes)
            
            print('Real frames shot: ', len(real_frames))
            temporal_ious_regions = []
            for idx, frame in enumerate(real_frames): #Detect persons frame by frame 
                print('Shot frame to process: ',idx)
                persons_in_frame = []
                persons_filtered = []
                if type_person_detector == constants.YOLO:
                    img_size = 416
                    conf_thres = 0.8
                    nms_thres = 0.4
                    persons_in_frame = localization_utils.personDetectionInFrameYolo(person_model, img_size, conf_thres, nms_thres, classes, frame)
                elif type_person_detector == constants.MASKRCNN:
                    mask_rcnn_threshold = 0.4
                    persons_in_frame = MaskRCNN.personDetectionInFrameMaskRCNN(person_model, frame, mask_rcnn_threshold)
                print('--num Persons in frame: ', len(persons_in_frame))
                
                if len(persons_in_frame) > 0:
                    persons_in_segment.append(persons_in_frame)
                    iou_threshold = 0.3
                    thres_intersec_person_saliency = 0.5
                    thresh_close_persons = 20
                
                    if bbox_last is not None:
                        print('='*10, ' ONLINE ')
                        dynamic_region = saliency_bboxes[0]
                        persons_filtered, only_joined_regions = localization_utils.filterClosePersonsInFrame(persons_in_frame, thresh_close_persons)
                        print('---Persons after filter close: ', len(persons_filtered), len(only_joined_regions))
                        # persons_segment_filtered.append(persons_filtered)
                        for person in persons_filtered:
                            iou = localization_utils.IOU(person, dynamic_region)
                            print('----IOU (person and dynamic region): ', str(iou))
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
                        print('---Persons after filter close: ', len(persons_filtered), len(only_joined_regions))
                        # persons_segment_filtered.append(persons_filtered)
                        
                        for personBox in persons_filtered:
                            for saliencyBox in saliency_bboxes:
                                iou = localization_utils.IOU(personBox, saliencyBox)
                                # tmp_rgion = localization_utils.joinBBoxes(saliencyBox,personBox) #Nooo si el saliente bbox es todo frame
                                tmp_rgion = personBox
                                tmp_rgion.score = iou
                                temporal_ious_regions.append(tmp_rgion)
                                print('----IOU (person and dynamic region): ', str(iou))
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

                            # print('anomalous_regions final: ', len(anomalous_regions))
                        ######################################              
                        
                        # l = [segmentBox]
                        # print('=========== Ground Truth Score =============')
                        # scoring.getScoresFromRegions(video_name[0], bbox_segments, l, classifier, transforms_dataset['test'])
                    
                        # for b in segment_real_info['real_bboxes']:
                        #     b.score = l[0].score
                    if len(anomalous_regions) > 0 :
                        break
                else:
                    persons_in_segment.append(None) #only to plot

            if len(anomalous_regions) == 0:
                print('Tracking Algorithm FAIL!!!!. Using last localization...')
                anomalous_regions.append(bbox_last)   
            
            # video_persons_in_segment.append(persons_in_frame)
            # video_persons_segment_filtered.append(persons_segment_filtered)
            # video_anomalous_regions.append(anomalous_regions)

            iou = localization_utils.IOU(segmentBox,anomalous_regions[0])
            row = [video_name[0]+'---segment No: '+str(num_segment), iou]
            data_rows.append(row)
            # data_rows_video.append(row)
            dis_images, segment_info, idx_next_segment = anomalyDataset.computeSegmentDynamicImg(idx_video=idx_video, idx_next_segment=idx_next_segment)
            bbox_last = anomalous_regions[0]        
            if plot:
                dynamic_image = ttttt
                # preprocess = preprocesing_outs[3]
                preprocesing_outs[0] = np.stack((preprocesing_outs[0],)*3, axis=-1)
                preprocesing_outs[1] = np.stack((preprocesing_outs[1],)*3, axis=-1)
                # print('preprocess: ', preprocesing_outs[0].shape, preprocesing_outs[1].shape, preprocesing_outs[2].shape, preprocesing_outs[3].shape)
                mascara = np.stack((mascara,)*3, axis=-1)
                preprocess = np.concatenate((mascara,preprocesing_outs[0], preprocesing_outs[1], preprocesing_outs[2], preprocesing_outs[3]), axis=1)
                plotOpencv(real_frames,real_bboxes, persons_in_segment, anomalous_regions, dynamic_image, mascara, preprocess, saliency_bboxes)
                ######################################
            
    # with open("videos_scores.txt", "w") as txt_file:
    #     for line in videos_scores:
    #         txt_file.write(" ".join(line) + "\n")

    # print(videos_scores)

    ############# MAP #################
    print('data rows: ', len(data_rows))
    df = pd.DataFrame(data_rows, columns=['path', 'iou'])
    export_csv = df.to_csv ('ious.csv', index = None, header=True)
    df['tp/fp'] = df['iou'].apply(lambda x: 'TP' if x >= 0.5 else 'FP')
    export_csv = df.to_csv ('initial.csv', index = None, header=True)
    prec_at_rec, avg_prec, df = localization_utils.mAP(df)
    export_csv = df.to_csv ('final.csv', index = None, header=True)
    print('11 point precision is ', prec_at_rec)
    print('\nmap is ', avg_prec)

    # if plot and video_name is not None:
    #     font_size = 9
    #     subplot_r = 2
    #     subplot_c = 5
    #     for i in range(num_video_segments):
    #         di_image = video_dynamic_images[i]
    #         preprocesing_reults = np.empty((subplot_r, subplot_c), dtype=tuple)
    #         preprocesing_outs = video_preprocesing_outs[i]
    #         mascara = video_mascaras[i]

    #         preprocesing_reults[0,0] = (mascara,'image source')
    #         preprocesing_reults[0,1]=(preprocesing_outs[0], 'thresholding')
    #         preprocesing_reults[0,2]=(preprocesing_outs[1], 'morpho')
    #         preprocesing_reults[0,3]=(preprocesing_outs[2], 'contours')
    #         preprocesing_reults[0, 4] = (preprocesing_outs[3], 'bboxes')
    #         real_frames = video_real_frames[i]
    #         real_bboxes = video_real_bboxes[i]
    #         saliency_bboxes = video_saliency_bboxes[i]
    #         persons_in_segment = video_persons_in_segment[i]
    #         persons_segment_filtered = video_persons_segment_filtered[i]
    #         anomalous_regions = video_anomalous_regions[i]
    #         myplot(subplot_r, subplot_c, font_size, preprocesing_reults, di_image, real_frames, real_bboxes, saliency_bboxes, persons_in_segment,
    #                 persons_segment_filtered, anomalous_regions)

    

def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saliencyModelFile", type=str)
    parser.add_argument("--batchSize", type=int, default=3)
    # parser.add_argument("--numEpochs", type=int, default=10)
    parser.add_argument("--numWorkers", type=int, default=1)
    parser.add_argument("--numDiPerVideos", type=int, default=5)
    parser.add_argument("--shuffle", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--plot", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--videoSegmentLength", type=int, default=15)
    parser.add_argument("--personDetector", type=str, default=constants.YOLO)
    parser.add_argument("--positionSegment", type=str)
    parser.add_argument("--overlapping", type=float)
    parser.add_argument("--videoName", type=str, default=None)

    args = parser.parse_args()
    plot = args.plot
    maxNumFramesOnVideo = 0
    videoSegmentLength = args.videoSegmentLength
    typePersonDetector = args.personDetector
    numDiPerVideos = args.numDiPerVideos
    positionSegment = args.positionSegment
    overlapping = args.overlapping
    num_classes = 2 #anomalus or not
    input_size = (224,224)
    transforms_dataset = transforms_anomaly.createTransforms(input_size)
    dataset_source = 'frames'
    batch_size = args.batchSize
    num_workers = args.numWorkers
    saliency_model_file = args.saliencyModelFile
    shuffle = args.shuffle
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    threshold = 0.5
    video_name = args.videoName

    saliency_model_config = saliency_model_file
    getRawFrames = True
    path_dataset = constants.PATH_UCFCRIME2LOCAL_FRAMES_REDUCED
    train_videos_path = os.path.join(constants.PATH_UCFCRIME2LOCAL_README, 'Train_split_AD.txt')
    test_videos_path = os.path.join(constants.PATH_UCFCRIME2LOCAL_README, 'Test_split_AD.txt')
    dataloaders_dict, test_names, anomalyDataset = anomalyInitializeDataset.initialize_final_only_test_anomaly_dataset(path_dataset, train_videos_path,
                                                        test_videos_path, batch_size, num_workers, numDiPerVideos, transforms_dataset,
                                                        maxNumFramesOnVideo, videoSegmentLength, positionSegment, shuffle, getRawFrames,
                                                        overlapping)

    saliency_tester = saliencyTester.SaliencyTester(saliency_model_file, num_classes, dataloaders_dict['test'], test_names,
                                        input_size, saliency_model_config, numDiPerVideos, threshold)
    
    #classifierFile = '/media/david/datos/PAPERS-SOURCE_CODE/violencedetection/ANOMALYCRIME/checkpoints/resnet18_Finetuned-False-_di-2_fusionType-tempMaxPool_num_epochs-20_videoSegmentLength-30_positionSegment-random-FINAL.pth'
    # classifierFile = '/media/david/datos/PAPERS-SOURCE_CODE/violencedetection/ANOMALYCRIME/checkpoints/resnet18_Finetuned-False-_di-3_fusionType-maxTempPool_num_epochs-23_videoSegmentLength-30_positionSegment-begin-FINAL.pth'
    # classifier = torch.load(classifierFile)
    # classifier.eval()
    # classifier.inferenceMode()
    print(torch.__version__)
    h = 240
    w = 320
    # offline(dataloaders_dict['test'], saliency_tester, typePersonDetector, h, w, plot)
    online(anomalyDataset, saliency_tester, typePersonDetector, h, w, plot, video_name)
    # df = pd.read_csv('final.csv')
    # print(df.head(10))
    # df = df.sort_values('iou',ascending=False)
    # df = df.reset_index(drop=True)
    # print(df.head(10))
    # # print(len(df.index))

    # prec_at_rec, avg_prec, df = localization_utils.mAPPascal(df)
    # export_csv = df.to_csv ('finalmAPPascal.csv', index = None, header=True)
    # print(df.head(10))

    
    # raw_size = (h, w)
    # resize2raw_transform = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.Resize(raw_size),
    #     transforms.ToTensor()
    #     ])

    # resize2cnn_transform = transforms.Compose(
    #     [
    #         transforms.ToPILImage(),
    #         transforms.Resize((224,224)),
    #         transforms.ToTensor(),
    #         # transforms.Normalize([0.49237782, 0.49160805, 0.48998737], [0.11053326, 0.11088469, 0.11275752] )
    #         # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    #     ]
    # )
    # crop_transform = transforms.Compose(
    #     [
    #         transforms.ToPILImage(),
    #         transforms.CenterCrop((224,224)),
    #         transforms.ToTensor(),
    #         # transforms.Normalize([0.49237782, 0.49160805, 0.48998737], [0.11053326, 0.11088469, 0.11275752] )
    #         # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    #     ]
    # )
    
    
    # Set up formatting for the movie files
    # Writer = animation.writers['ffmpeg']
    # writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    # type_set_frames = constants.FRAME_POS_ALL
    # first = 0
    
    # fig, ax = plt.subplots()    
   
__main__()


