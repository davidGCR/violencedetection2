
import sys
sys.path.insert(1, '/media/david/datos/PAPERS-SOURCE_CODE/violencedetection')
import argparse
import ANOMALYCRIME.transforms_anomaly as transforms_anomaly
import ANOMALYCRIME.anomaly_initializeDataset as anomaly_initializeDataset
import SALIENCY.saliencyTester as saliencyTester
# import SALIENCY
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

def maskRCNN():
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model
 
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

    args = parser.parse_args()
    plot = args.plot
    maxNumFramesOnVideo = 0
    videoSegmentLength = args.videoSegmentLength
    personDetector = args.personDetector
    numDiPerVideos = args.numDiPerVideos
    positionSegment = 'begin'
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

    saliency_model_file = args.saliencyModelFile
    saliency_model_config = saliency_model_file

    path_dataset = constants.PATH_UCFCRIME2LOCAL_FRAMES_REDUCED
    train_videos_path = os.path.join(constants.PATH_UCFCRIME2LOCAL_README, 'Train_split_AD.txt')
    test_videos_path = os.path.join(constants.PATH_UCFCRIME2LOCAL_README, 'Test_split_AD.txt')
    dataloaders_dict, test_names = anomaly_initializeDataset.initialize_final_only_test_anomaly_dataset(path_dataset, train_videos_path, test_videos_path, dataset_source, batch_size,
                                                        num_workers, numDiPerVideos, transforms_dataset, maxNumFramesOnVideo, videoSegmentLength, positionSegment, shuffle)
    tester = saliencyTester.SaliencyTester(saliency_model_file, num_classes, dataloaders_dict['test'], test_names,
                                        input_size, saliency_model_config, numDiPerVideos, threshold)
    img_size = 416
    weights_path = "YOLOv3/weights/yolov3.weights"
    class_path = "YOLOv3/data/coco.names"
    model_def = "YOLOv3/config/yolov3.cfg"
    conf_thres = 0.8
    nms_thres = 0.4
    yolo_model, classes = yolo_inference.initializeYoloV3(img_size, class_path, model_def, weights_path)
    h = 240
    w = 320
    raw_size = (h, w)
    resize_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(raw_size),
        transforms.ToTensor()
        ])
    data_rows = []
    mask_model = maskRCNN()
    # Set up formatting for the movie files
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    
    # fig, ax = plt.subplots()


    for i, data in enumerate(dataloaders_dict['test'], 0):
        print("-" * 150)
        #di_images = [1,ndis,3,224,224]
        dis_images, labels, video_name, bbox_segments = data
        print(video_name, dis_images.size(), len(bbox_segments))
        bbox_segments = np.array(bbox_segments)
        ######################## dynamic images
        l_di_masked = []
        l_di_images = []
        dis_images = dis_images.detach().cpu()
        for di_image in dis_images:
            l_di_images.append(di_image)

        ######################## mask
        masks = tester.compute_mask(dis_images, labels)
        masks = torch.squeeze(masks, 0)  #tensor [ndis,1,224,224]
        masks = masks.detach().cpu()
        l_masks = []
        
        for mask in masks:
            mask = resize_transform(mask.cpu())
            mask = tester.normalize_tensor(mask)
            mask = tensor2numpy(mask)
            l_masks.append(mask)
            # saliency_bboxes, preprocesing_reults = localization_utils.computeBoundingBoxFromMask(masks)  #only one by segment
            

        ######################## dynamic images masked
        dis_masked = dis_images.detach().cpu() * masks.detach().cpu()  #tensor [1,3,224,224]
        dis_masked = torch.squeeze(dis_masked, 0)  #tensor [3,224,224]
        dis_masked = dis_masked.detach().cpu()

        video_prepoc_saliencies = []
        for di_masked in dis_masked: 
            di_masked = resize_transform(di_masked)
            di_masked = tester.normalize_tensor(di_masked)
            di_masked = tensor2numpy(di_masked)
            l_di_masked.append(di_masked)
            saliency_bboxes, preprocesing_reults = localization_utils.computeBoundingBoxFromMask(di_masked)  #only one by segment
            video_prepoc_saliencies.append({
                'saliency bboxes': saliency_bboxes,
                'preprocesing': preprocesing_reults
            })
        
        print('video_prepoc_saliencies: ', len(video_prepoc_saliencies))
    #     ########################### To plot masks
    #     shape = masks.shape
    #     if shape[2] == 1:
    #         masks = np.squeeze(masks,2)
    #         masks = localization_utils.gray2rgbRepeat(masks)
        
    #     # ######################### To plot a dynamic image
    #     # di_image = torch.squeeze(di_images, 0)
    #     # di_image = di_image / 2 + 0.5      
    #     # di_image = resize_transform(di_image)
    #     # di_image = di_image.numpy()
    #     # di_image = np.transpose(di_image, (1, 2, 0))
    #     # print('di_image numpy', di_image.shape)

    #     # # di_image_gray = cv2.cvtColor(di_image, cv2.COLOR_BGR2GRAY)
    #     # # di_image_t, preprocesing_reults_di = localization_utils.computeBoundingBoxFromMask(di_image_gray)

    #     # fig, axes = plt.subplots(2, 5, figsize=(15, 5))
    #     # print('axes: ', type(axes), axes.shape)
    #     # im0 = axes[0,0].imshow(masks)
    #     # im1 = axes[0,1].imshow(preprocesing_reults['mask'])
    #     # im2 = axes[0,2].imshow(preprocesing_reults['process_mask'])
    #     # im3 = axes[0,3].imshow(preprocesing_reults['contours'])
    #     # im4 = axes[0,4].imshow(preprocesing_reults['boxes'])
        
    #     # im5 = axes[1,0].imshow(di_masked)
    #     # im6 = axes[1,1].imshow(preprocesing_reults_di['mask'])
    #     # im7 = axes[1,2].imshow(preprocesing_reults_di['process_mask'])
    #     # im8 = axes[1,3].imshow(preprocesing_reults_di['contours'])
    #     # im9 = axes[1,4].imshow(preprocesing_reults_di['boxes'])
    #     # plt.show()
        video_real_info = []
        for bbox_segment in bbox_segments:
            # print('bbox_segment: ',bbox_segment)
            #read frames of segment
            frames_names, real_frames, real_bboxes = localization_utils.getFramesFromSegment(video_name[0], bbox_segment, 'all')
            video_real_info.append({
                'frames_names':frames_names,
                'real_frames':real_frames,
                'real_bboxes':real_bboxes})
            # print('real_frames, real_bboxes: ', len(frames_names), len(real_frames), len(real_bboxes))
        print('video_real_info: ', len(video_real_info))
        type_set_frames = constants.FRAME_POS_ALL
        first = 0
        # end = len(real_frames) - 1
        # area_threshold = 40
        # persons_distance_threshold = 15
        
        mask_rcnn_threshold = 0.3
        
        
        font_size = 10

        for index, segment_real_info in enumerate(video_real_info): #di by di
            #person detection
            persons_in_segment = []
            anomalous_regions = []  # to plot
            persons_in_frame = []
            persons_segment_filtered = [] #only to vizualice
            for idx, frame in enumerate(segment_real_info['real_frames']):
                if personDetector == constants.YOLO:
                    persons_in_frame = localization_utils.personDetectionInFrame(yolo_model, img_size, conf_thres, nms_thres, classes, frame)
                    # print('persons_in_frame MaskRCNN: ', len(persons_in_frame))
                elif personDetector == constants.MASKRCNN:
                    persons_in_frame = MaskRCNN.personDetectionInFrameMaskRCNN(mask_model, frame, mask_rcnn_threshold)
                print('num Persons in frame: ', len(persons_in_frame))
                persons_in_segment.append(persons_in_frame)
        
            #refinement in segment
            for i,persons_in_frame in enumerate(persons_in_segment):
                persons_filtered, only_joined_regions = localization_utils.filterClosePersonsInFrame(persons_in_frame, 20)
                persons_segment_filtered.append(only_joined_regions)
                # print('persons_in_frame filter: ', len(persons_filtered))
                anomalous_regions_in_frame = localization_utils.findAnomalyRegionsOnFrame(persons_filtered, video_prepoc_saliencies[index]['saliency bboxes'], 0.3)
                # anomalous_regions_in_frame = sorted(anomalous_regions_in_frame, key=lambda x: x.iou, reverse=True)
                
                anomalous_regions.extend(anomalous_regions_in_frame)
                print('--------------- anomalous_regions: ', len(anomalous_regions))
                anomalous_regions.sort(key=lambda x: x.iou, reverse=True) #sort by iuo to get only one anomalous regions
                            
            if len(anomalous_regions) == 0:
                # print('No anomalous regions found...')
                iou = localization_utils.IOU(segment_real_info['real_bboxes'][0],None)
            else:
                iou = localization_utils.IOU(segment_real_info['real_bboxes'][0],anomalous_regions[0])
            row = [frames_names[first], iou]
            data_rows.append(row)
        
    #     if plot:           
    #         # create a figure with two subplots
    #         fig, axes = plt.subplots(1, 5,figsize=(25,5))
    #         def f(image):
    #             return np.array(image)

            
    #         ######################### To plot a dynamic image
    #         di_image = torch.squeeze(di_images, 0)
    #         di_image = di_image / 2 + 0.5      
    #         di_image = resize_transform(di_image)
    #         di_image = di_image.numpy()
    #         di_image = np.transpose(di_image, (1, 2, 0))
    #         ims = []

    #         # img_process_mask = localization_utils.gray2rgbRepeat(preprocesing_reults['process_mask'])

    #         di_image = localization_utils.plotOnlyBBoxOnImage(di_image, saliency_bboxes, constants.PIL_YELLOW, 'saliency', font_size)
    #         image_anomalous, image_anomalous_final = None, None
    #         for i in range(len(real_frames)):
    #             # real_frame = real_frames[i].copy()
    #             gt_image = localization_utils.plotOnlyBBoxOnImage(real_frames[i].copy(), real_bboxes[i], constants.PIL_RED, 'gtruth', font_size)
    #             # image = localization_utils.plotOnlyBBoxOnImage(di_images, real_bboxes[i], constants.PIL_RED, 'Ground Truth')
    #             if len(saliency_bboxes) > 0:
    #                 image_saliency = localization_utils.plotOnlyBBoxOnImage(gt_image, saliency_bboxes, constants.PIL_YELLOW, 'saliency',font_size)
    #                 image_persons = localization_utils.plotOnlyBBoxOnImage(image_saliency, persons_in_segment[i], constants.PIL_GREEN, 'person', font_size)
                    
                    
    #                 image_persons_filt = localization_utils.plotOnlyBBoxOnImage(real_frames[i].copy(), persons_segment_filtered[i], constants.PIL_MAGENTA, 'person_filter',font_size)
    #                 image_anomalous = localization_utils.plotOnlyBBoxOnImage(real_frames[i].copy(), anomalous_regions, constants.PIL_BLUE, 'anomalous', font_size)
    #                 if len(anomalous_regions)>0:
    #                     image_anomalous = localization_utils.plotOnlyBBoxOnImage(image_anomalous, anomalous_regions[0], constants.PIL_MAGENTA, 'anomalous', font_size)
                                       
    #             im0 = axes[0].imshow(f(masks))
    #             im1 = axes[1].imshow(f(di_masked))
    #             im2 = axes[2].imshow(f(image_persons))
    #             im3 = axes[3].imshow(f(image_persons_filt))
    #             im4 = axes[4].imshow(f(image_anomalous))
                
    #             ims.append([im0])
    #             ims.append([im1])
    #             ims.append([im2])
    #             ims.append([im3])
    #             ims.append([im4])
    #         print('ims: ', len(ims))
    #         ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=100)
    #         # ani.save('RESULTS/animations/animation.mp4', writer=writer)          
            
    #         plt.show()
    #         # ani.save('RESULTS/animations/animation.gif', writer='imagemagick', fps=30)
    # ############# MAP #################
    print('data rows: ', len(data_rows))
    df = pd.DataFrame(data_rows, columns=['path', 'iou'])
    df['tp/fp'] = df['iou'].apply(lambda x: 'TP' if x >= 0.5 else 'FP')
    localization_utils.mAP(df)
   
__main__()

# while (1):
        #     plt.show()
        #     # cv2.imshow('eefef', img)
        #     k = cv2.waitKey(33)
        #     if k == -1:
        #         continue
        #     if k == ord('a'):
        #         break
        #     if k == ord('q'):
        #         # localization_utils.tuple2BoundingBox(bboxes[0])
        #         sys.exit('finish!!!') 