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
from torchvision.utils import make_grid

def __main__ ():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifierFile", type=str)
    parser.add_argument("--batchSize", type=int, default=3)
    # parser.add_argument("--numEpochs", type=int, default=10)
    parser.add_argument("--numWorkers", type=int, default=1)
    parser.add_argument("--numDiPerVideos", type=int, default=5)
    parser.add_argument("--shuffle", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--plot", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--videoSegmentLength", type=int, default=15)

    args = parser.parse_args()
    plot = args.plot
    maxNumFramesOnVideo = 0
    videoSegmentLength = args.videoSegmentLength
    numDiPerVideos = args.numDiPerVideos
    positionSegment = 'random'
    num_classes = 2 #anomalus or not
    input_size = (224,224)
    transforms_dataset = transforms_anomaly.createTransforms(input_size)
    batch_size = args.batchSize
    num_workers = args.numWorkers
    classifierFile = args.classifierFile
    shuffle = args.shuffle
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    threshold = 0.5

    path_dataset = constants.PATH_UCFCRIME2LOCAL_FRAMES_REDUCED
    train_videos_path = os.path.join(constants.PATH_UCFCRIME2LOCAL_README, 'Train_split_AD.txt')
    test_videos_path = os.path.join(constants.PATH_UCFCRIME2LOCAL_README, 'Test_split_AD.txt')
    dataloaders_dict, test_names = anomaly_initializeDataset.initialize_final_only_test_anomaly_dataset(path_dataset, train_videos_path, test_videos_path,
                                    batch_size, num_workers, numDiPerVideos, transforms_dataset, maxNumFramesOnVideo, videoSegmentLength, positionSegment, shuffle)
    
    h = 240
    w = 320
    raw_size = (h, w)
    resize_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(raw_size),
        transforms.ToTensor()
        ])

    classifier = torch.load(classifierFile)
    classifier.eval()
    classifier.inferenceMode()


    for i, data in enumerate(dataloaders_dict['test'], 0):
        print("-" * 150)
        #di_images = [1,ndis,3,224,224]
        dis_images, labels, video_name, bbox_segments = data
        print(video_name, dis_images.size(), len(bbox_segments))
        
        dis_images = dis_images.to(device)
        labels = labels.to(device)

        # dis_images = torch.stack([dis_images, dis_images], dim=0)
        print('dis images stacked: ', dis_images.size())

        with torch.set_grad_enabled(False):
            outputs = classifier(dis_images)
            # p = torch.nn.functional.softmax(outputs, dim=1)
            values, indices = torch.max(outputs, 1)

        print(' ******* LABELS: ',labels, ', OUTPUTS: ', outputs,', VALUES: ',values, ', INDICES: ', indices)
        # print(labels,outputs, values, indices)

        # bbox_segments = np.array(bbox_segments)
        # ######################## dynamic images
        # l_source_frames = []
        # l_di_images = [] # to plot
        # dis_images = dis_images.detach().cpu()
        # dis_images = torch.squeeze(dis_images, 0) ## to num dynamic images > 1 and minibatch == 1
        # for di_image in dis_images:
        #     # di_image = di_image / 2 + 0.5  q    
        #     # di_image = resize_transform(di_image)
        #     di_image = di_image.numpy()
        #     di_image = np.transpose(di_image, (1, 2, 0))
        #     l_di_images.append(di_image)
__main__()