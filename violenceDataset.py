from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import cv2
import torch
import glob
import time
from dynamicImage import *
import random


class ViolenceDataset(Dataset):
    def __init__(self, dataset, labels, numFrames, spatial_transform, numDynamicImagesPerVideo, videoSegmentLength, positionSegment):
        self.spatial_transform = spatial_transform
        self.videos = dataset
        self.labels = labels
        self.numFrames = numFrames
        self.numDynamicImagesPerVideo = numDynamicImagesPerVideo
        self.videoSegmentLength = videoSegmentLength
        self.positionSegment = positionSegment

    def __len__(self):
        return len(self.videos)

    def skip_initial_segments(self, percentage, video_splits_by_no_Di):
        cut = int((len(video_splits_by_no_Di)*percentage)/100) 
        # print('Trimed sequence: ', len(sequences), cut)
        video_splits_by_no_Di = video_splits_by_no_Di[cut:]
        return video_splits_by_no_Di, cut

    def getBeginSegment(self, frames_list, skip_frames):
        cut = int((len(frames_list)*skip_frames)/100) 
        segment = frames_list[cut:cut + self.videoSegmentLength]
        return segment

    def getRandomSegment(self, video_splits_by_no_Di, idx): #only if numDiPerVideo == 1
        random_segment = None
        label = int(self.labels[idx])
        random_segment_idx = random.randint(0, len(video_splits_by_no_Di) - 1)
        random_segment = video_splits_by_no_Di[random_segment_idx]
        if self.numFrames[idx] > self.videoSegmentLength:
            if label == 0: #Normal videos
                video_splits_by_no_Di, _ = self.skip_initial_segments(self.skipPercentage,video_splits_by_no_Di) #skip 35% of initial segments
            while len(random_segment) != self.videoSegmentLength:
                random_segment_idx = random.randint(0, len(video_splits_by_no_Di) - 1)
                random_segment = video_splits_by_no_Di[random_segment_idx]
        # print('random sequence:', random_segment_idx, len(random_segment))
        return random_segment
     
    def getCentralSegment(self, video_splits_by_no_Di, idx): #only if numDiPerVideo == 1
        segment = None
        central_segment_idx = int(len(video_splits_by_no_Di)/2) 
        segment = video_splits_by_no_Di[central_segment_idx]
        return segment

    def getVideoSegments(self, vid_name, idx):
        frames_list = os.listdir(vid_name)
        frames_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
        video_segments = []
        seqLen = 0 #number of frames for each segment
        
        if self.numFrames[idx] <= self.videoSegmentLength:
            seqLen = self.numFrames[idx]
            # print('Short video: ', vid_name, self.numFrames[idx], 'seqLen:', seqLen)
        else:
            seqLen = self.videoSegmentLength
        num_frames_on_video = self.numFrames[idx]
        # print('range: ', 'vid name: ', vid_name,seqLen,'numFrames: ', self.numFrames[idx], 'segmentLenght: ', self.videoSegmentLength)
        video_splits_by_no_Di = [frames_list[x:x + seqLen] for x in range(0, num_frames_on_video, seqLen)]

        if len(video_splits_by_no_Di) > 1:
            last_idx = len(video_splits_by_no_Di)-1
            split = video_splits_by_no_Di[last_idx]
            if len(split) < 10:
                del video_splits_by_no_Di[last_idx]
                
        if len(video_splits_by_no_Di) < self.numDynamicImagesPerVideo:
            diff = self.numDynamicImagesPerVideo - len(video_splits_by_no_Di)
            last_idx = len(video_splits_by_no_Di) - 1
            for i in range(diff):
                video_splits_by_no_Di.append(video_splits_by_no_Di[last_idx])

        if self.numDynamicImagesPerVideo == 1:
            if self.positionSegment == 'random':
                segment = self.getRandomSegment(video_splits_by_no_Di, idx)
            elif self.positionSegment == 'central':
                segment = self.getCentralSegment(video_splits_by_no_Di, idx)
            elif self.positionSegment == 'begin':
                self.skipPercentage = 0
                segment = self.getBeginSegment(frames_list, self.skipPercentage)
            video_segments = []
            video_segments.append(segment)
        elif self.numDynamicImagesPerVideo > 1:
            # print('hereeeeeeeeeeee> ', self.numDynamicImagesPerVideo)
            for i in range(len(video_splits_by_no_Di)):
                if i < self.numDynamicImagesPerVideo:
                    video_segments.append(video_splits_by_no_Di[i])
                else:
                    break
                
        return video_segments, seqLen

    def __getitem__(self, idx):
        vid_name = self.videos[idx]
        # print(vid_name)
        label = self.labels[idx]
        dinamycImages = []
        sequences, seqLen = self.getVideoSegments(vid_name, idx) # bbox_segments: (1, 16, 6)= (no segments,no frames segment,info
        for seq in sequences:
            frames = []
            # r_frames = []
            for frame in seq:
                img_dir = str(vid_name) + "/" + frame
                img1 = Image.open(img_dir).convert("RGB")
                img = np.array(img1)
                frames.append(img)
            imgPIL, img = getDynamicImage(frames)
            imgPIL = self.spatial_transform(imgPIL.convert("RGB"))
            dinamycImages.append(imgPIL)
            
        dinamycImages = torch.stack(dinamycImages, dim=0)  #torch.Size([bs, ndi, ch, h, w])
        if self.numDynamicImagesPerVideo == 1:
            dinamycImages = dinamycImages.squeeze(dim=0) ## get normal pytorch tensor [bs, ch, h, w]
        
        
        return dinamycImages, label, vid_name, 0#dinamycImages, label:  <class 'torch.Tensor'> <class 'int'> torch.Size([3, 224, 224])

    



