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
import re

class AnomalyOnlineDataset(Dataset):
    def __init__(self, dataset, labels, numFrames, bbox_files, spatial_transform, videoBlockLength, numDynamicImgsPerBlock,
                    videoSegmentLength, overlappingBlock, overlappingSegment):
        self.spatial_transform = spatial_transform
        self.videos = dataset
        self.labels = labels
        self.numFrames = numFrames  #dataset total num frames by video
        self.bbox_files = bbox_files  # dataset bounding box txt file associated to video
        
        self.numDynamicImgsPerBlock = numDynamicImgsPerBlock  #number of segments from the video/ -1 means all the segments from the video
        self.videoSegmentLength = videoSegmentLength # max number of frames by segment video
        
        # self.maxNumFramesOnVideo = maxNumFramesOnVideo # to use only some frames
        # self.positionSegment = positionSegment  #could be at begin, central or random
        self.skipPercentage = 35
        # self.getRawFrames = getRawFrames
        self.overlappingSegment = overlappingSegment
        self.overlappingBlock = overlappingBlock
        self.videoBlockLength = videoBlockLength

    def __len__(self):
        return len(self.videos)

    def skip_initial_segments(self, percentage, video_splits_by_no_Di):
        cut = int((len(video_splits_by_no_Di)*percentage)/100) 
        # print('Trimed sequence: ', len(sequences), cut)
        video_splits_by_no_Di = video_splits_by_no_Di[cut:]
        return video_splits_by_no_Di, cut

    def getOneSegment(self, frames_list, start):
        if start == 0: #first segment
            segment = frames_list[start:start + self.videoSegmentLength]
            idx_next_segment = self.videoSegmentLength
        else:
            start = start - int(self.videoSegmentLength * self.overlappingSegment)
            end = start + self.videoSegmentLength
            segment = frames_list[start:end]
            idx_next_segment = end
        
        return segment, idx_next_segment

    def getOneBlock(self, frames_list, start):
        segments_block = []
        if start == 0: #first segment
            block = frames_list[start:start + self.videoBlockLength]
            idx_next_block = self.videoBlockLength
            
        else:
            start = start - int(self.videoBlockLength * self.overlappingBlock)
            end = start + self.videoBlockLength
            block = frames_list[start:end]
            idx_next_block = end
            
        
        if self.numDynamicImgsPerBlock > 1:
            # print('Len block: ', len(block))
            seqLen = int(len(block)/self.numDynamicImgsPerBlock)
            segments_block = [block[x:x + seqLen] for x in range(0, len(block), seqLen)]
        else:
            segments_block.append(block)
        return block, idx_next_block, segments_block
  
    def getSegmentInfo(self, segment, bdx_file_path):
        data = []
        if bdx_file_path is not None:
            with open(bdx_file_path, 'r') as file:
                for row in file:
                    data.append(row.split())
            data = np.array(data)
        segment_info=[]
        for frame in segment:
            splits = re.split('(\d+)', frame)
            num_frame = int(splits[1])
            if num_frame != int(data[num_frame, 5]):
                sys.exit('Houston we have a problem: index frame does not equal to the bbox file!!!')
            
            flac = int(data[num_frame,6]) # 1 if is occluded: no plot the bbox
            xmin = int(data[num_frame, 1])
            ymin= int(data[num_frame, 2])
            xmax = int(data[num_frame, 3])
            ymax = int(data[num_frame, 4])
            # print(type(frame), type(flac), type(xmin), type(ymin))
            info_frame = [frame, flac, xmin, ymin, xmax, ymax]
            
            segment_info.append(info_frame)
        # print(segment_info)
        return segment_info

    
    def computeBlockDynamicImg(self, idx_video, idx_next_block):
        vid_name = self.videos[idx_video]
        dinamycImages = []
        frames_list = os.listdir(vid_name)
        frames_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
        bdx_file_path = self.bbox_files[idx_video]
        if idx_next_block < self.numFrames[idx_video]:
            block, idx_next_block, segments_block = self.getOneBlock(frames_list, idx_next_block)
            one_segment = []
            for segment in segments_block:
                one_segment.extend(segment)
            
            block_boxes_info = self.getSegmentInfo(one_segment,bdx_file_path)
            # print('segments_block: ', segments_block, idx_next_block)
            for segment in segments_block:
                frames = []
                for frame in segment:
                    img_dir = str(vid_name) + "/" + frame
                    img1 = Image.open(img_dir).convert("RGB")
                    img = np.array(img1)
                    frames.append(img)
                imgPIL, img = getDynamicImage(frames)
                imgPIL = self.spatial_transform(imgPIL.convert("RGB"))
                dinamycImages.append(imgPIL)
            dinamycImages = torch.stack(dinamycImages, dim=0)  #torch.Size([bs, ndi, ch, h, w])
            # if self.numDynamicImgsPerBlock == 1:
            #     dinamycImages = dinamycImages.squeeze(dim=0)
            return dinamycImages, idx_next_block, block_boxes_info
        else:
            return None, None

    def computeSegmentDynamicImg(self, idx_video, idx_next_segment):
        vid_name = self.videos[idx_video]
        bbox_file = self.bbox_files[idx_video]
        dinamycImages = []
        frames_list = os.listdir(vid_name)
        frames_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
        if idx_next_segment < self.numFrames[idx_video]:
            segment, idx_next_segment = self.getOneSegment(frames_list, idx_next_segment)
            segment_info =  self.getSegmentInfo(segment, bbox_file)
            frames = []
            for frame in segment:
                img_dir = str(vid_name) + "/" + frame
                img1 = Image.open(img_dir).convert("RGB")
                img = np.array(img1)
                frames.append(img)

            imgPIL, img = getDynamicImage(frames)
            imgPIL = self.spatial_transform(imgPIL.convert("RGB"))
            dinamycImages.append(imgPIL)
                
            dinamycImages = torch.stack(dinamycImages, dim=0)  #torch.Size([bs, ndi, ch, h, w])
            
            dinamycImages = dinamycImages.squeeze(dim=0)  ## get normal pytorch tensor [bs, ch, h, w]
            return dinamycImages, segment_info, idx_next_segment
        else:
            return None, None, None

    def __getitem__(self, idx):
        vid_name = self.videos[idx]
        # print(vid_name)
        label = self.labels[idx]
        
        return vid_name, label
    
    def getindex(self, vid_name):
        matching = [s for s in self.videos if vid_name in s]
        
        if len(matching)>0:
            vid_name = matching[0]
            index = self.videos.index(vid_name)
            print('ONLY video: ',matching,index)
            return index
        else:
            return None
        # # print(vid_name)
        # label = self.labels[idx]
        
        # return vid_name, label




