from torch.utils.data import Dataset
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


class AnomalyDataset(Dataset):
    def __init__(self, dataset, labels, numFrames, bbox_files, spatial_transform, nDynamicImages,
                    videoSegmentLength, maxNumFramesOnVideo, positionSegment):
        self.spatial_transform = spatial_transform
        self.images = dataset
        self.labels = labels
        self.numFrames = numFrames  #dataset total num frames by video
        self.bbox_files = bbox_files  # dataset bounding box txt file associated to video
        
        self.numDynamicImagesPerVideo = nDynamicImages  #number of segments from the video/ -1 means all the segments from the video
        self.videoSegmentLength = videoSegmentLength # max number of frames by segment video
        
        self.maxNumFramesOnVideo = maxNumFramesOnVideo # to use only some frames
        self.positionSegment = positionSegment  #could be at begin, central or random
        self.skipPercentage = 35 

    def __len__(self):
        return len(self.images)

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
    
    def get_bbox_segmet(self, video_segments, bdx_file_path, idx):
        data = []
        bbox_segments = [] #all video [[info, info, ...] , [info, info, ...], ...] ##info is a list: [num_frame, flac, xmin, ymin, xmax, ymax]
        if bdx_file_path is not None:
            with open(bdx_file_path, 'r') as file:
                for row in file:
                    data.append(row.split())
            data = np.array(data)
            for segment in video_segments:
                bbox_infos_frames = []
                for frame in segment:
                    # print('frame: ', frame)
                    num_frame = int(frame[len(frame) - 7:-4])
                    if num_frame != int(data[num_frame, 5]):
                        sys.exit('Houston we have a problem: index frame does not equal to the bbox file!!!')
                        # print('Houston we have a problem: index frame does not equal to the bbox file!!!')
                    
                    flac = int(data[num_frame,6]) # 1 if is occluded: no plot the bbox
                    xmin = int(data[num_frame, 1])
                    ymin= int(data[num_frame, 2])
                    xmax = int(data[num_frame, 3])
                    ymax = int(data[num_frame, 4])
                    # print(type(frame), type(flac), type(xmin), type(ymin))
                    info_frame = [frame, flac, xmin, ymin, xmax, ymax]
                    bbox_infos_frames.append(info_frame)
                bbox_segments.append(bbox_infos_frames)
        return bbox_segments

    def getVideoSegments(self, vid_name, idx):
        frames_list = os.listdir(vid_name)
        frames_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
        bbox_file = self.bbox_files[idx]
        video_segments = []
        seqLen = 0 #number of frames for each segment
        
        # if self.numFrames[idx]  < self.videoSegmentLength * self.numDynamicImagesPerVideo:
        #     seqLen = self.numFrames[idx] // self.numDynamicImagesPerVideo
        if self.numFrames[idx] <= self.videoSegmentLength:
            seqLen = self.numFrames[idx]
            # print('Short video: ', vid_name, self.numFrames[idx], 'seqLen:', seqLen)
        else:
            seqLen = self.videoSegmentLength
        num_frames_on_video = self.numFrames[idx] 
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
                
        bbox_segments =  self.get_bbox_segmet(video_segments, bbox_file, idx)
        return video_segments, seqLen, bbox_segments

    def __getitem__(self, idx):
        vid_name = self.images[idx]
        # print(vid_name)
        label = self.labels[idx]
        dinamycImages = []
        sequences, seqLen, bbox_segments = self.getVideoSegments(vid_name, idx) # bbox_segments: (1, 16, 6)= (no segments,no frames segment,info
        # print('SEGmentos lenght: ', len(sequences), vid_name)
        for seq in sequences:
            frames = []
            # print('segment lenght: ', len(seq))
            for frame in seq:
                img_dir = str(vid_name) + "/" + frame
                img = Image.open(img_dir).convert("RGB")
                img = np.array(img)
                frames.append(img)
            imgPIL, img = getDynamicImage(frames)
            imgPIL = self.spatial_transform(imgPIL.convert("RGB"))
            dinamycImages.append(imgPIL)
            
        dinamycImages = torch.stack(dinamycImages, dim=0)  #torch.Size([bs, ndi, ch, h, w])
        # print('dynamic images: ', dinamycImages.size())
        if self.numDynamicImagesPerVideo == 1:
            dinamycImages = dinamycImages.squeeze(dim=0) ## get normal pytorch tensor [bs, ch, h, w]
        return dinamycImages, label, vid_name, bbox_segments

    def getVideoSegments_old(self, vid_name, idx):
        frames_list = os.listdir(vid_name)
        frames_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
        bbox_file = self.bbox_files[idx]
        video_segments = []
        seqLen = 0 #number of frames for each segment
        if self.numDynamicImagesPerVideo == 1 and self.videoSegmentLength > 0:
            if self.numFrames[idx] <= self.videoSegmentLength:
                # print('hereeeeeeeeeeeeeee....')
                seqLen = self.numFrames[idx]
            else:
                seqLen = self.videoSegmentLength
            num_frames_on_video = self.numFrames[idx] 
            video_splits_by_no_Di = [frames_list[x:x + seqLen] for x in range(0, num_frames_on_video, seqLen)]
            if self.positionSegment == 'random':
                segment = self.getRandomSegment(video_splits_by_no_Di, idx)
            elif self.positionSegment == 'central':
                segment = self.getCentralSegment(video_splits_by_no_Di, idx)
            elif self.positionSegment == 'begin':
                self.skipPercentage = 0
                segment = self.getBeginSegment(frames_list, self.skipPercentage)
            video_segments = []
            video_segments.append(segment)
        else:
            if self.maxNumFramesOnVideo == 0: #use all frames from video
                num_frames_on_video = self.numFrames[idx]
            else: #use some frames from video
                num_frames_on_video = self.maxNumFramesOnVideo if self.numFrames[idx] >= self.maxNumFramesOnVideo else self.numFrames[idx]
            seqLen = num_frames_on_video // self.numDynamicImagesPerVideo  # calculate num frames by segment
            # print('num_frames_on_video , segment length: ', num_frames_on_video, seqLen)
            cut = 0
            if self.labels[idx] == 0: # normal videos
                video_segments, cut = self.skip_initial_segments(self.skipPercentage, video_segments)  #skip 35% of initial segments
            video_segments = [frames_list[x:x + seqLen] for x in range(cut, num_frames_on_video + cut, seqLen)]
            if len(video_segments) > self.numDynamicImagesPerVideo: #cut last segments
                diff = len(video_segments) - self.numDynamicImagesPerVideo
                video_segments = video_segments[: - diff]
            # print(len(video_segments))
        bbox_segments =  self.get_bbox_segmet(video_segments, bbox_file, idx)
        return video_segments, seqLen, bbox_segments



