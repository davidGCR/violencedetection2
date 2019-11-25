
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


 
def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saliencyModelFile", type=str)
    parser.add_argument("--batchSize", type=int, default=3)
    # parser.add_argument("--numEpochs", type=int, default=10)
    parser.add_argument("--numWorkers", type=int, default=1)
    parser.add_argument("--numDiPerVideos", type=int, default=5)
    parser.add_argument("--shuffle", type=lambda x: (str(x).lower() == 'true'), default=False)

    args = parser.parse_args()
    maxNumFramesOnVideo = 0
    videoSegmentLength = 10
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
        saliency_bboxes = localization_utils.computeBoundingBoxFromMask(masks)
        
        #read frames of segment
        type_set_frames = 'all'
        real_frames,  real_bboxes = localization_utils.getFramesFromSegment(video_name[0], bbox_segments[0], type_set_frames)
        print('real_frames, real_bboxes: ', len(real_frames), len(real_bboxes))
        # for bbox in real_bboxes:
        #     print(bbox)
        bbox_persons_in_segment = localization_utils.personDetectionInSegment(real_frames, yolo_model,
                                                                            img_size, conf_thres, nms_thres, classes, 'first')

        # print('bbox_persons_in_segment: ', len(bbox_persons_in_segment), len(bbox_persons_in_segment[0]))
        anomalous_regions = localization_utils.findAnomalyRegionsOnFrame(bbox_persons_in_segment[0], saliency_bboxes, 48, 50)
        print('anomalous_regions', type(anomalous_regions), len(anomalous_regions))
        
        shape = masks.shape
        if shape[2] == 1:
            masks = np.squeeze(masks,2)
            masks = localization_utils.gray2rgbRepeat(masks)

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
            if len(anomalous_regions)>0:
                image = localization_utils.plotOnlyBBoxOnImage(image, anomalous_regions[0], constants.PIL_WHITE, 'Anomalous')
            im = plt.imshow(f(image), animated=True)
            # fig, ax = localization_utils.plotBBoxesOnImage(fig, ax, violent_regions, constants.CYAN, 'anomalous')
            # ax = localization_utils.plotBBoxesOnImage(ax, violent_regions, constants.CYAN,'anomalous')
            ims.append([im])
        ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=100)              
        plt.show()
       
        # fig2 = plt.figure(figsize=(15., 4.))
        # fig2, ax2 = plt.subplots()
        
            # ax2 = localization_utils.plotBBoxesOnImage(ax2, violent_regions, constants.CYAN,'anomalous')
        # localization_utils.plot_grid(fig2, debug_frames)
            # plt.show()
            # plt.pause(0.0005)
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
    # saliencyTester.test(saliency_model_file, num_classes, dataloaders_dict['test'], test_names, input_size, saliency_model_config, numDiPerVideos)

__main__()