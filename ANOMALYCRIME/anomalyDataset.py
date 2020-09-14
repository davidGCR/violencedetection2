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
import torchvision.transforms as transforms
from torch.utils.data._utils.collate import default_collate

class AnomalyDataset(Dataset):
    def __init__(self, dataset, labels, numFrames, bbox_files, spatial_transform, nDynamicImages,
                    videoSegmentLength, positionSegment, overlaping, frame_skip):
        self.spatial_transform = spatial_transform
        self.images = dataset
        self.labels = labels
        self.numFrames = numFrames  #dataset total num frames by video
        self.bbox_files = bbox_files  # dataset bounding box txt file associated to video
        
        self.numDynamicImagesPerVideo = nDynamicImages  #number of segments from the video/ -1 means all the segments from the video
        self.videoSegmentLength = videoSegmentLength # max number of frames by segment video
        
        # self.maxNumFramesOnVideo = maxNumFramesOnVideo # to use only some frames
        self.positionSegment = positionSegment  #could be at begin, central or random
        self.skipPercentage = 35
        # self.getRawFrames = getRawFrames
        self.overlaping = overlaping
        self.frame_skip = frame_skip

    def __len__(self):
        return len(self.images)

    def skip_initial_segment(self, video_segments):
        if len(video_segments) > 1:  #Normal videos
            if len(video_segments[len(video_segments)-1]) < 0.8 * self.videoSegmentLength:
                video_segments.pop(len(video_segments)-1)
            else:
                video_segments.pop(0)
        return video_segments

    def getRandomSegment(self, video_segments, int_label): #only if numDiPerVideo == 1
        random_segment = None
        # label = int(self.labels[idx])
        if label == 0 and len(video_segments)>1:  #Normal videos
            video_segments.pop(0)
        random_segment_idx = random.randint(0, len(video_segments) - 1)
        random_segment = video_segments[random_segment_idx]
       
        return random_segment
     
    def getCentralSegment(self, video_segments): #only if numDiPerVideo == 1
        segment = None
        central_segment_idx = int(len(video_segments)/2) 
        segment = video_segments[central_segment_idx]
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

    def padding(self, video_segments):
        last_segment = video_segments[len(video_segments)-1]
        while len(video_segments) < self.numDynamicImagesPerVideo:
            video_segments.append(last_segment)
        return video_segments

    def getVideoSegments(self, vid_name, idx):
        frames_list = os.listdir(vid_name)
        frames_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
        # bbox_file = self.bbox_files[idx]
        video_segments = []
        if self.videoSegmentLength == -1:
            seqLen = self.numFrames[idx]
        else:
            seqLen = self.videoSegmentLength  #number of frames for each segment
        
        indices = [x for x in range(0, self.numFrames[idx], self.frame_skip + 1)]
        indices_segments = [indices[x:x + seqLen] for x in range(0, len(indices), seqLen)]
        for i, indices_segment in enumerate(indices_segments):
            segment = np.asarray(frames_list)[indices_segment].tolist()
            video_segments.append(segment)
            
        num_frames_on_video = self.numFrames[idx]
        if self.numDynamicImagesPerVideo == 1:
            if self.positionSegment == 'random':
                label = self.labels[idx]
                segment = self.getRandomSegment(video_segments, label)
            elif self.positionSegment == 'central':
                segment = self.getCentralSegment(video_segments)
            elif self.positionSegment == 'begin':
                label = self.labels[idx]
                if label == 0 :  #Normal videos
                    video_segments = self.skip_initial_segment(video_segments)
                segment = video_segments[0]
            video_segments = []
            video_segments.append(segment)
        else:
            if len(video_segments) < self.numDynamicImagesPerVideo:
                video_segments = self.padding(video_segments)
            elif len(video_segments) > self.numDynamicImagesPerVideo:
                label = self.labels[idx]
                if label == 0 :  #Normal videos
                    video_segments = self.skip_initial_segment(video_segments)
                while len(video_segments) > self.numDynamicImagesPerVideo:
                    video_segments.pop(len(video_segments)-1)
                    
        return video_segments, indices_segments

    def __getitem__(self, idx):
        vid_name = self.images[idx]
        label = self.labels[idx]
        # print('================= ',vid_name, label, self.numFrames[idx])
        dinamycImages = []
        sequences, indices_segments = self.getVideoSegments(vid_name, idx)  # bbox_segments: (1, 16, 6)= (no segments,no frames segment,info
        len_segments = [len(seg) for seg in indices_segments]
        for seq in sequences:
            
            # print('original num segments: {}, segment len {}, segment{}'.format(len(indices_segments), len(seq), seq))
            # print(seq)
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
            if self.spatial_transform is not None:
                imgPIL = self.spatial_transform(imgPIL.convert("RGB"))
            # else:
            #     imgPIL = transforms.ToTensor()(imgPIL)
            dinamycImages.append(imgPIL)
            
        dinamycImages = torch.stack(dinamycImages, dim=0)  #torch.Size([bs, ndi, ch, h, w])
        if self.numDynamicImagesPerVideo == 1:
            dinamycImages = dinamycImages.squeeze(dim=0) ## get normal pytorch tensor [bs, ch, h, w]
        return dinamycImages, label, vid_name, 0

    


# print('SEGmentos lenght: ', len(sequences), vid_name)
        # video_raw_frames = []
        # for seq in sequences:
        #     frames = []
        #     for frame in seq:
        #         img_dir = str(vid_name) + "/" + frame
        #         img = Image.open(img_dir).convert("RGB")
        #         frames.append(img)
        #         img = np.array(img)
        #         img = torch.from_numpy(img).float()
                
        #         # print('---', img.size())
        #     if self.getRawFrames:
        #         # video_raw_frames.append(torch.stack(frames,0))
        #         video_raw_frames.append(frames)
        #     imgPIL, img = computeDynamicImage(torch.stack(frames, dim=0))
            
        #     imgPIL = self.spatial_transform(imgPIL.convert("RGB"))
        #     dinamycImages.append(imgPIL)
        # print(len(sequences))
