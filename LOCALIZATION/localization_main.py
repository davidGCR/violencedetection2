
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

    args = parser.parse_args()
    plot = args.plot
    maxNumFramesOnVideo = 0
    videoSegmentLength = 30
    numDiPerVideos = args.numDiPerVideos
    positionSegment = 'begin'
    num_classes = 2 #anomalus or not
    input_size = 224
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

    for i, data in enumerate(dataloaders_dict['test'], 0):
        print("-" * 150)
        di_images, labels, video_name, bbox_segments = data
        print(video_name, len(bbox_segments))
        bbox_segments = np.array(bbox_segments)
        # print('bbox_segments: ', bbox_segments.shape)  #(1, 16, 6)
        
        masks = tester.compute_mask(di_images, labels)
        masks = torch.squeeze(masks, 0) #tensor [1,224,224]
        masks = resize_transform(masks.cpu())
        # print('masks: ', type(masks), masks.size())
        masks = masks.detach().cpu()
        masks = tester.normalize_tensor(masks)
        # masks = masks.repeat(3, 1, 1)
        masks = tensor2numpy(masks)
        # print('masks numpy', masks.shape)
        saliency_bboxes = localization_utils.computeBoundingBoxFromMask(masks) #only one by segment
        
        #read frames of segment
        frames_names, real_frames,  real_bboxes = localization_utils.getFramesFromSegment(video_name[0], bbox_segments[0], 'all')
        print('real_frames, real_bboxes: ', len(frames_names), len(real_frames), len(real_bboxes))
        # for bbox in real_bboxes:
        #     print(bbox)
        type_set_frames = 'all'
        # bbox_persons_in_segment = localization_utils.personDetectionInSegment(real_frames, yolo_model,
        #                                                                     img_size, conf_thres, nms_thres, classes, type_set_frames)
        # anomalous_regions = localization_utils.findAnomalyRegionsOnFrame(bbox_persons_in_segment[0], saliency_bboxes, 48, 50)
        
        # print('anomalous_regions', type(anomalous_regions), len(anomalous_regions))
        first = 0
        end = len(real_frames) - 1
        area_threshold = 40
        persons_distance_threshold = 15
        anomalous_regions = []  # to plot
        mask_rcnn_threshold = 0.3
        if type_set_frames == constants.FRAME_POS_FIRST:
            #person detection
            # persons_in_frame = localization_utils.personDetectionInFrame(yolo_model, img_size, conf_thres, nms_thres, classes, real_frames[first])
            persons_in_frame = MaskRCNN.personDetectionInFrameMaskRCNN(mask_model, real_frames[first], mask_rcnn_threshold)
            # print('persons: ', persons_in_frame)
            #refinement
            anomalous_region = localization_utils.findAnomalyRegionsOnFrame(persons_in_frame, saliency_bboxes, area_threshold, persons_distance_threshold)
            anomalous_regions.append(anomalous_region)
            # print('ÍOU', type(real_bboxes[first]), type(anomalous_region))
            iou = localization_utils.IOU(real_bboxes[first],anomalous_region)
            row = [frames_names[first], iou]
            data_rows.append(row)
        elif type_set_frames == constants.FRAME_POS_EXTREMES:
            #person detection
            # persons_in_frame = localization_utils.personDetectionInFrame(yolo_model, img_size, conf_thres, nms_thres, classes, real_frames[first])
            persons_in_frame = MaskRCNN.personDetectionInFrameMaskRCNN(mask_model, real_frames[first], mask_rcnn_threshold)
            #refinement
            anomalous_region = localization_utils.findAnomalyRegionsOnFrame(persons_in_frame, saliency_bboxes, area_threshold, persons_distance_threshold)
            anomalous_regions.append(anomalous_region)
            iou = localization_utils.IOU(real_bboxes[first],anomalous_region)
            row = [frames_names[first], iou]
            data_rows.append(row)

            #person detection
            # persons_in_frame = localization_utils.personDetectionInFrame(yolo_model, img_size, conf_thres, nms_thres, classes, real_frames[end])
            persons_in_frame = MaskRCNN.personDetectionInFrameMaskRCNN(mask_model, real_frames[first], mask_rcnn_threshold)
            #refinement
            anomalous_region = localization_utils.findAnomalyRegionsOnFrame(persons_in_frame, saliency_bboxes, area_threshold, persons_distance_threshold)
            anomalous_regions.append(anomalous_region)
            iou = localization_utils.IOU(real_bboxes[end],anomalous_region)
            row = [frames_names[end], iou]
            data_rows.append(row)
        elif type_set_frames == constants.FRAME_POS_ALL:
            for idx, frame in enumerate(real_frames):
                #person detection
                # persons_in_frame = localization_utils.personDetectionInFrame(yolo_model, img_size, conf_thres, nms_thres, classes, frame)
                persons_in_frame = MaskRCNN.personDetectionInFrameMaskRCNN(mask_model, real_frames[first], mask_rcnn_threshold)
                print('persons_in_frame MaskRCNN: ', len(persons_in_frame))
                #refinement
                anomalous_region = localization_utils.findAnomalyRegionsOnFrame(persons_in_frame, saliency_bboxes, area_threshold, persons_distance_threshold)
                anomalous_regions.append(anomalous_region)
                iou = localization_utils.IOU(real_bboxes[idx],anomalous_region)
                row = [frames_names[first], iou]
                data_rows.append(row)
        
        if plot:
            shape = masks.shape
            if shape[2] == 1:
                masks = np.squeeze(masks,2)
                masks = localization_utils.gray2rgbRepeat(masks)
                # masks = (masks * 255).astype(np.uint8)
                # masks = Image.fromarray(masks, mode='L')
            # di_images = torch.squeeze(di_images, 0)
            # di_images = di_images / 2 + 0.5
            # di_transform = transforms.Compose([
            #                     transforms.ToPILImage(),
            #                     transforms.Resize(raw_size),
            #                     # transforms.ToTensor()
            #                     ])
            # di_images = di_transform(di_images)
            # print('di shape: ', type(di_images))
            fig, ax = plt.subplots()
            # ax.imshow(real_frames[len(real_frames)//2])
            # ax.imshow(masks)
            # ax = localization_utils.plotBBoxesOnImage(ax, saliency_bboxes, constants.RED, 'saliency')
            # ax = localization_utils.plotBBoxesOnImage(ax, bbox_persons_in_segment[0], constants.GREEN, 'person')
            # ax = localization_utils.plotBBoxesOnImage(ax, violent_regions, constants.CYAN,'anomalous')
            # plt.show()

            def f(image):
                return np.array(image)
            ims = []
            for i in range(len(real_frames)):            
                image = localization_utils.plotOnlyBBoxOnImage(real_frames[i], real_bboxes[i], constants.PIL_RED, 'Ground Truth')
                # image = localization_utils.plotOnlyBBoxOnImage(di_images, real_bboxes[i], constants.PIL_RED, 'Ground Truth')
                if len(anomalous_regions) > 0:
                    image = localization_utils.plotOnlyBBoxOnImage(image, anomalous_regions[i], constants.PIL_WHITE, 'Anomalous')
                im = plt.imshow(f(image), animated=True)
                # fig, ax = localization_utils.plotBBoxesOnImage(fig, ax, violent_regions, constants.CYAN, 'anomalous')
                # ax = localization_utils.plotBBoxesOnImage(ax, violent_regions, constants.CYAN,'anomalous')
                ims.append([im])
            ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=100)
            # ani.save('RESULTS/animations/'+str(i)+'.mp4', writer=writer)          
            ani.save('RESULTS/animations/'+str(i)+'.gif', writer='imagemagick', fps=30)
            plt.show()
    # print('data rows: ', len(data_rows))
    df = pd.DataFrame(data_rows, columns=['path', 'iou'])
    df['tp/fp'] = df['iou'].apply(lambda x: 'TP' if x >= 0.5 else 'FP')
    localization_utils.mAP(df)
    # print(df.head(10))
   
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