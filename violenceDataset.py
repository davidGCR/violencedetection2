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
    def __init__(self, dataset, labels, numFrames, spatial_transform, numDynamicImagesPerVideo,
                                videoSegmentLength, positionSegment, overlaping, frame_skip):
        self.spatial_transform = spatial_transform
        self.videos = dataset
        self.labels = labels
        self.numFrames = numFrames
        self.numDynamicImagesPerVideo = numDynamicImagesPerVideo
        self.videoSegmentLength = videoSegmentLength
        self.positionSegment = positionSegment
        self.overlaping = overlaping
        self.frame_skip = frame_skip

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

    def getVideoSegmentsOverlapped(self, vid_name, idx):
        frames_list = os.listdir(vid_name)
        frames_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
        video_segments = []
        seqLen = self.videoSegmentLength
        o = int(self.overlaping * seqLen)
        indices = [x for x in range(0, self.numFrames[idx], self.frame_skip + 1)]
        indices_segments = [indices[x:x + seqLen] for x in range(0, len(indices), seqLen-o)]
        
        for indices_segment in indices_segments:
            video_segments.append(frames_list[indices_segment])

        # video_segments.append(indices[0:seqLen])
        # i = seqLen
        # end = 0
        # while i < len(indices):
        #     if len(video_segments) == self.numDynamicImagesPerVideo:
        #         break
        #     else:
        #         start = i - num_frames_overlapped #10 20 
        #         end = start + seqLen  #29 39
        #         i = end #29 39
        #         video_segments.append(indices[start:end])
               
        return video_segments
        
    def getVideoSegments(self, vid_name, idx):
        frames_list = os.listdir(vid_name)
        frames_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
        video_segments = []
        seqLen = self.videoSegmentLength
        indices = [x for x in range(0, self.numFrames[idx], self.frame_skip + 1)]
        
        video_splits_by_no_Di = []
        indices_segments = [indices[x:x + seqLen] for x in range(0, len(indices), seqLen)]
        for indices_segment in indices_segments:
            video_splits_by_no_Di.append(frames_list[indices_segment])

        if len(video_splits_by_no_Di) > 1:
            last_idx = len(video_splits_by_no_Di)-1
            split = video_splits_by_no_Di[last_idx]
            if len(split) < 10:
                del video_splits_by_no_Di[last_idx]
                

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
                
        return video_segments

    def __getitem__(self, idx):
        vid_name = self.videos[idx]
        # print(vid_name)
        label = self.labels[idx]
        dinamycImages = []
        # print('jujujujujuj:', self.overlaping)
        if self.overlaping > 0:
            
            sequences = self.getVideoSegmentsOverlapped(vid_name, idx)
            # print(len(sequences), self.numDynamicImagesPerVideo)
            # print(sequences)
        else:
            # print('fdsgfjsdhgkjshgksdgs')
            sequences = self.getVideoSegments(vid_name, idx) # bbox_segments: (1, 16, 6)= (no segments,no frames segment,info
        
        preprocessing_time = 0.0
        for seq in sequences:
            # print(len(seq))
            frames = []
            # r_frames = []
            for frame in seq:
                img_dir = str(vid_name) + "/" + frame
                img1 = Image.open(img_dir).convert("RGB")
                img = np.array(img1)
                frames.append(img)
            start_time = time.time()
            imgPIL, img = getDynamicImage(frames)
            end_time = time.time()
            imgPIL = self.spatial_transform(imgPIL.convert("RGB"))
            
            preprocessing_time += (end_time - start_time)
            dinamycImages.append(imgPIL)
        # print('Len: ', len(dinamycImages), vid_name)
        dinamycImages = torch.stack(dinamycImages, dim=0)  #torch.Size([bs, ndi, ch, h, w])
        if self.numDynamicImagesPerVideo == 1:
            dinamycImages = dinamycImages.squeeze(dim=0) ## get normal pytorch tensor [bs, ch, h, w]
        
        
        return dinamycImages, label, vid_name, preprocessing_time #dinamycImages, label:  <class 'torch.Tensor'> <class 'int'> torch.Size([3, 224, 224])


class ViolenceDatasetAumented(Dataset):
    def __init__(self, images, labels, spatial_transform):
        self.spatial_transform = spatial_transform
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        # print(vid_name)
        label = self.labels[idx]        
        imgPIL = Image.open(image_path).convert("RGB")
        imgPIL = self.spatial_transform(imgPIL.convert("RGB"))
        # dinamycImages.append(imgPIL)
            
        # dinamycImages = torch.stack(dinamycImages, dim=0)  #torch.Size([bs, ndi, ch, h, w])
        # if self.numDynamicImagesPerVideo == 1:
        #     dinamycImages = dinamycImages.squeeze(dim=0) ## get normal pytorch tensor [bs, ch, h, w]
        
        
        return imgPIL, label, image_path, 0 #dinamycImages, label:  <class 'torch.Tensor'> <class 'int'> torch.Size([3, 224, 224])



