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
import torch.nn.functional as F

def pytorch_show(img):
    # img = img / 2 + 0.5 
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')

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
    
    # h = 240
    # w = 320
    h = 224
    w = 224
    raw_size = (h, w)
    resize_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(raw_size),
        transforms.ToTensor(),
        transforms.Normalize([0.49237782, 0.49160805, 0.48998737], [0.11053326, 0.11088469, 0.11275752] )
        ])

    classifier = torch.load(classifierFile)
    classifier.eval()
    classifier.inferenceMode()

    unfold = torch.nn.Unfold(kernel_size=(32, 32),stride=32)
    # Pad tensor to get the same output
    
    kh, kw = 32, 32
    dh, dw = 32, 32
    # ax.axis("off")
    
    for i, data in enumerate(dataloaders_dict['test'], 0):
        print("-" * 150)
        #di_images = [1,ndis,3,224,224]
        dis_images, labels, video_name, bbox_segments = data
        print(video_name)
        
        dis_images = dis_images.to(device)
        labels = labels.to(device)
        print('dis images: ', dis_images.size())
        ############### PLOT ##################
        di_image = torch.squeeze(dis_images,dim=0)
        di_image = di_image.cpu().numpy()
        di_image = np.transpose(di_image, (1, 2, 0))
        # print('di_image: ',di_image.shape)
        di_image = di_image / 2 + 0.5
        fig, ax = plt.subplots(figsize = (10,10))
        ax.imshow(di_image)
        plt.show()        

        # patches = unfold(dis_images) #patches:  torch.Size([1, 3072, 49]) : 30172=32x32x3, 49 patches: 7x7
        # patches = torch.squeeze(patches, dim=0)
        # print('----- patches: ', patches.size())
        # view = patches.view(49, 3, 32, 32)
        # view = view / 2 + 0.5
        # patches = patches.permute(2,0,1)
        dis_images = F.pad(dis_images, (1, 1, 1, 1))
        patches = dis_images.unfold(2, kh, dh).unfold(3, kw, dw)
        # View as [batch_size, height, width, channels*kh*kw]
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(*patches.size()[:3], -1)
        
        patches = torch.unsqueeze(patches, dim=0)
        patches = patches.view(49, 3, 32, 32)

        l_batch = []
        for j in range(49):
            patch = resize_transform(patches[j].cpu())
            # print('--patch: ',patch.size())
            l_batch.append(patch)
        
    
        
        batch = torch.stack(l_batch, dim=0)
        batch = batch.to(device)
        print('batch : ', batch.size())
        # ########### PLOT patches ###########
        
        patches = patches / 2 + 0.5
        pytorch_show(make_grid(patches.cpu().data, nrow=7, padding=10))
        plt.show()
        scores = []
        with torch.set_grad_enabled(False):
            outputs = classifier(dis_images)
            
            values, indices = torch.max(outputs, 1)

            outputs_patches = classifier(batch)
            p = torch.nn.functional.softmax(outputs_patches, dim=1)
            print('p>: ',p.size())
            scores.extend(p.cpu().numpy())
            values_patches, indices_patches = torch.max(outputs_patches, 1)


        print(' ******* LABELS: ',labels, ', OUTPUTS: ', outputs,', VALUES: ',values, ', INDICES: ', indices)
        print('patches output: ',outputs_patches.size(),', INDICES: ',indices_patches)

        for idx, img_patch in enumerate(l_batch):
            print('patch: ', idx, ', label: ', str(indices_patches[idx].data))
            print('val: ', str(values_patches[idx].data), ', score: ', str(p[idx].data))
            # img_patch = img_patch / 2 + 0.5
            img_patch = torch.unsqueeze(img_patch, dim=0)
            pytorch_show(make_grid(img_patch.data, nrow=7, padding=10))
            plt.show()
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