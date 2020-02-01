# from torch.utils.data import Dataset
import sys
# import include
# sys.path.insert(1,'/Users/davidchoqueluqueroman/Desktop/PAPERS-CODIGOS/violencedetection2')
sys.path.insert(1, '/media/david/datos/PAPERS-SOURCE_CODE/violencedetection')
from PIL import Image
import numpy as np
import os
# import cv2
import torch
import glob
import time
from dynamicImage import *
import random
import sys
import torchvision.transforms as transforms
import datasetUtils
import constants
from torch.utils.data import Dataset
# from torch.utils.data._utils.collate import default_collate

class DataAumentation(Dataset):
    def __init__(self, videos, labels, numFrames, spatial_transform,
                    videoSegmentLength, folder_output_path):
        self.spatial_transform = spatial_transform
        self.videos = videos
        self.labels = labels
        self.numFrames = numFrames  #dataset total num frames by video
        # self.bbox_files = bbox_files  # dataset bounding box txt file associated to video
        self.videoSegmentLength = videoSegmentLength # max number of frames by segment video
    
        self.skipPercentage = 35
        self.folder_output_path = folder_output_path
        # self.getRawFrames = getRawFrames
        # self.overlapping = overlapping

    def __len__(self):
        return len(self.videos)

    def skip_initial_segments(self, percentage, video_splits_by_no_Di):
        cut = int((len(video_splits_by_no_Di)*percentage)/100) 
        # print('Trimed sequence: ', len(sequences), cut)
        video_splits_by_no_Di = video_splits_by_no_Di[cut:]
        return video_splits_by_no_Di, cut



    def getVideoSegments(self, vid_name, idx):
        frames_list = os.listdir(vid_name)
        frames_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
        video_segments = []
        seqLen = 0 #number of frames for each segment
        
        # print('range: ', 'vid name: ', vid_name,seqLen,'numFrames: ', self.numFrames[idx], 'segmentLenght: ', self.videoSegmentLength)
        video_segments = []
        for x in range(0, self.numFrames[idx], self.videoSegmentLength):
            rr = x + self.videoSegmentLength
            if rr < self.numFrames[idx]:
                segment = frames_list[x:x + self.videoSegmentLength]
                video_segments.append(segment)
        
        return video_segments

    def __getitem__(self, idx):
        vid_name = self.videos[idx]
        # print(vid_name)
        # label = self.labels[idx]
        # dinamycImages = []
        video_segments = self.getVideoSegments(vid_name, idx) # bbox_segments: (1, 16, 6)= (no segments,no frames segment,info
        # # print('SEGmentos lenght: ', len(sequences), vid_name)
        # video_raw_frames = []

        for idx, seq in enumerate(video_segments):
            frames = []
            # r_frames = []
            for frame in seq:
                img_dir = str(vid_name) + "/" + frame
                img1 = Image.open(img_dir).convert("RGB")
                img = np.array(img1)
                frames.append(img)
            # if self.getRawFrames:
            #     video_raw_frames.append(frames)
            imgPIL, img = getDynamicImage(frames)
            # imgPIL = self.spatial_transform(imgPIL.convert("RGB"))
            head, tail = os.path.split(vid_name)
            folder_out = os.path.join(self.folder_output_path, tail)
            if not os.path.exists(folder_out):
                os.makedirs(folder_out)
            imgPIL.save(os.path.join(folder_out,str(idx)+'.png'))

            # dinamycImages.append(imgPIL)
        return vid_name
            
        # dinamycImages = torch.stack(dinamycImages, dim=0)  #torch.Size([bs, ndi, ch, h, w])
        # if self.numDynamicImagesPerVideo == 1:
        #     dinamycImages = dinamycImages.squeeze(dim=0) ## get normal pytorch tensor [bs, ch, h, w]
        # return dinamycImages, label, vid_name, bbox_segments

    
def __main__():
    path_dataset = constants.PATH_UCFCRIME2LOCAL_FRAMES_REDUCED
    train_videos_path = os.path.join(constants.PATH_UCFCRIME2LOCAL_README, 'Train_split_AD.txt')
    test_videos_path = os.path.join(constants.PATH_UCFCRIME2LOCAL_README, 'Test_split_AD.txt')
    train_names, train_labels, train_num_frames, train_bbox_files, test_names, test_labels, test_num_frames, test_bbox_files = datasetUtils.train_test_videos(train_videos_path, test_videos_path, path_dataset)
    # combined_train_names, combined_val_names, train_labels, val_labels = train_test_split(combined, train_labels, stratify=train_labels, test_size=0.30)
    # train_names, train_num_frames, train_bbox_files = zip(*combined_train_names)
    # val_names, val_num_frames, val_bbox_files = zip(*combined_val_names)
    train_labels = datasetUtils.labels_2_binary(train_labels)
    # val_labels = datasetUtils.labels_2_binary(val_labels)
    # test_labels = datasetUtils.labels_2_binary(test_labels)
    # print(train_labels)
    # dataset = DataAumentation(videos, labels, numFrames, bbox_files, spatial_transform,
    #                 videoSegmentLength, output_path)
    # dataset = DataAumentation([os.path.join(constants.PATH_UCFCRIME2LOCAL_FRAMES_REDUCED,'Arrest002')], [1], [279], None,
    #                 10, constants.PATH_DATA_AUMENTATION_OUTPUT)
    num_frames_to_dynamic_images = 10
    dataset = DataAumentation(train_names, train_labels, train_num_frames, None,
                    num_frames_to_dynamic_images, constants.PATH_DATA_AUMENTATION_OUTPUT)
    for video_name in dataset:
        print('Processing: ', video_name, '...')
        
    # dataset.__getitem__(0)
__main__()



