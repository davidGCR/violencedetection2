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
import time

class AnomalyOnlineDataset(Dataset):
    def __init__(self, dataset, labels, numFrames, bbox_files, spatial_transform, videoBlockLength, numDynamicImgsPerBlock,
                    videoSegmentLength, overlappingBlock, overlappingSegment, temporal_gts=None):
        self.spatial_transform = spatial_transform
        self.videos = dataset
        self.labels = labels
        self.numFrames = numFrames  #dataset total num frames by video
        self.bbox_files = bbox_files  # dataset bounding box txt file associated to video
        self.temporal_gts = temporal_gts
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

    def incrementOneBlock(self, frames_list, start_current_block, increment, idx_next_block, maxLengthBlock):
        # segments_block = []
        if (idx_next_block+increment) > maxLengthBlock:
            start_current_block = idx_next_block
            incremented_block = frames_list[start_current_block:start_current_block + self.videoBlockLength]
            idx_next_block = start_current_block + self.videoBlockLength
        if start_current_block == 0: #first segment
            incremented_block = frames_list[0:self.videoBlockLength]
            idx_next_block = self.videoBlockLength
        else:
            incremented_block = frames_list[start_current_block:idx_next_block + increment]
            idx_next_block = start_current_block + len(incremented_block)
        return incremented_block, start_current_block, idx_next_block
    
    def getOneBlock(self, frames_list, start, skip):
        segments_block = []
        # start = start
        # end = 0
        if start == 0: #first segment
            block = []
            # if self.videoSegmentLength > 0:

            for i in range(start,start + self.videoBlockLength, skip):
                # print('FFFRFRFRFRFR: ', frames_list[i])
                block.append(frames_list[i])
            # block = frames_list[start:start + self.videoBlockLength]
            idx_next_block = self.videoBlockLength
            
        else:
            start = start - int(self.videoBlockLength * self.overlappingBlock)
            end = start + self.videoBlockLength
            # block = frames_list[start:end]
            block = []
            for i in range(start,end, skip):
                if i<len(frames_list):
                    block.append(frames_list[i])
            idx_next_block = end
            
        
        if self.numDynamicImgsPerBlock > 1:
            # print('Len block: ', len(block))
            if self.overlappingSegment == 0:
                seqLen = int(len(block)/self.numDynamicImgsPerBlock)
                segments_block = [block[x:x + seqLen] for x in range(0, len(block), seqLen)]
            else:
                seqLen = self.videoSegmentLength
                num_frames_overlapped = int(self.overlappingSegment * seqLen)
                segments_block.append(block[0:seqLen])
                i = seqLen
                end = 0
                while i < len(block):
                    if len(segments_block) == self.numDynamicImgsPerBlock:
                        break
                    else:
                        start = i - num_frames_overlapped #10 20 
                        end = start + seqLen  #29 39
                        i = end #29 39
                        segments_block.append(block[start:end])
        else:
            segments_block.append(block)
        # print('block: ',block)
        return block, start, idx_next_block, segments_block
  
    def getTemporalGroundTruth(self, idx_video, segment):
        gt = self.temporal_gts[idx_video]
        # print('gt: ', type(gt[0]), gt)
        gt_segment = []
        for frame in segment:
            frame_number = re.findall(r'\d+', frame)
            frame_number = int(frame_number[0])
            if frame_number >= gt[0] and frame_number <= gt[1] or frame_number >= gt[2] and frame_number <= gt[3]:
                gt_segment.append(1)
            else:
                gt_segment.append(0)
        return gt_segment

    def videoGroundTruth(self, idx_video):
        bdx_file_path = self.bbox_files[idx_video]
        video_gt = []
        vid_name = self.videos[idx_video]
        frames_list = os.listdir(vid_name)
        frames_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
        if bdx_file_path is not None:
            data = [] 
            with open(bdx_file_path, 'r') as file:
                for row in file:
                    data.append(row.split())
            data = np.array(data)
            
            for frame in frames_list:
                # frame = segment[i]
                # print('frame: ', frame)
                splits = re.split('(\d+)', frame)
                num_frame = int(splits[1])
                num_frame = num_frame - 1
                
                if num_frame != int(data[num_frame, 5]):
                    sys.exit('Houston we have a problem: index frame does not equal to the bbox file!!!')
                if num_frame < len(data):
                    flac = int(data[num_frame,6]) # 1 if is lost: no plot the bbox
                    xmin = int(data[num_frame, 1])
                    ymin= int(data[num_frame, 2])
                    xmax = int(data[num_frame, 3])
                    ymax = int(data[num_frame, 4])
                    # print(type(frame), type(flac), type(xmin), type(ymin))
                    info_frame = [frame, flac, xmin, ymin, xmax, ymax, num_frame]
                else:
                    info_frame = [frame, -1, -1, -1, -1, -1, num_frame]
                video_gt.append(info_frame)
        else:
            for frame in frames_list:
                info_frame = [frame, -1, -1, -1, -1, -1, num_frame]
                video_gt.append(info_frame)
        return video_gt
    
    def getSegmentGt(self, video_gt, start, end):
        
        if start > 0:
            start = start + int(self.videoBlockLength * self.overlappingBlock)
        # print('----start:%d, end:%d'%(start,end))
        gt = video_gt[start:end]
        return gt



    def getSegmentInfo(self, segment, bdx_file_path):
        data = []
        segment_info=[]
        if bdx_file_path is not None:
            with open(bdx_file_path, 'r') as file:
                for row in file:
                    data.append(row.split())
            data = np.array(data)
            for frame in segment:
                # frame = segment[i]
                # print('frame: ', frame)
                splits = re.split('(\d+)', frame)
                num_frame = int(splits[1])
                num_frame = num_frame - 1
                
                if num_frame != int(data[num_frame, 5]):
                    sys.exit('Houston we have a problem: index frame does not equal to the bbox file!!!')
                if num_frame < len(data):
                    flac = int(data[num_frame,6]) # 1 if is lost: no plot the bbox
                    xmin = int(data[num_frame, 1])
                    ymin= int(data[num_frame, 2])
                    xmax = int(data[num_frame, 3])
                    ymax = int(data[num_frame, 4])
                    # print(type(frame), type(flac), type(xmin), type(ymin))
                    info_frame = [frame, flac, xmin, ymin, xmax, ymax]
                else:
                    info_frame = [frame, -1, -1, -1, -1, -1]
                segment_info.append(info_frame)
        else:
            # print('fgmdsofjhdfopjhodjh')
            for frame in segment:
                info_frame = [frame, -1, -1, -1, -1, -1]
                segment_info.append(info_frame)
        # print(segment_info)
        return segment_info

    
    def computeIncrementalBlockDynamicImg(self, idx_video, start_current_block, num_frames_increment, max_block_len, idx_next_block, skip):
        vid_name = self.videos[idx_video]
        dinamycImages = []
        frames_list = os.listdir(vid_name)
        frames_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
        if idx_next_block < self.numFrames[idx_video]:
            block, idx_next_block, segments_block = self.getOneBlock(frames_list, idx_next_block, skip)
            idx_next_block = idx_next_block + num_frames_increment
            if len(block) < max_block_len and idx_next_block<len(frames_list):
                block.append(frames_list[idx_next_block:idx_next_block+num_frames_increment])
                # i = i + num_frames_increment  

    def computeBlockDynamicImg(self, idx_video, idx_next_block, skip):
        vid_name = self.videos[idx_video]
        dinamycImages = []
        frames_list = os.listdir(vid_name)
        frames_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
        video_gt = self.videoGroundTruth(idx_video)
        # print('idx_next_block:', idx_next_block)
        if idx_next_block < self.numFrames[idx_video]:
            # start = idx_next_block
            block, start, idx_next_block, segments_block = self.getOneBlock(frames_list, idx_next_block,skip)
            # print('start:%d, end:%d'%(start,idx_next_block))
            
            block_gt = self.getSegmentGt(video_gt, start, idx_next_block)
            # one_segment = [] #join segments
            # for segment in segments_block:
            #     one_segment.extend(segment)
            # if self.temporal_gts is not None:
            #     block_boxes_info = self.getTemporalGroundTruth(idx_video,one_segment)
            # else:
            #     bdx_file_path = self.bbox_files[idx_video]
            #     block_boxes_info = self.getSegmentInfo(one_segment,bdx_file_path)

            
            # print(len(segments_block))
            for segment in segments_block:
                # print(len(segment))
                # print(segment)
                frames = []
                for i in range(len(segment)):
                    frame = segment[i]
                    img_dir = str(vid_name) + "/" + frame
                    img1 = Image.open(img_dir).convert("RGB")
                    img = np.array(img1)
                    frames.append(img)
                start_time = time.time()
                imgPIL, img = getDynamicImage(frames)
                end_time = time.time()
                spend_time = end_time - start_time
                # print('Dynamic image time: ', spend_time)
                imgPIL = self.spatial_transform(imgPIL.convert("RGB"))
                dinamycImages.append(imgPIL)
            dinamycImages = torch.stack(dinamycImages, dim=0)  #torch.Size([bs, ndi, ch, h, w])
            # if self.numDynamicImgsPerBlock == 1:
            #     dinamycImages = dinamycImages.squeeze(dim=0)
            return dinamycImages, idx_next_block, block_gt, spend_time
        else:
            return None, None, None, 0

    def computeSegmentDynamicImg(self, idx_video, idx_next_segment):
        vid_name = self.videos[idx_video]
        bbox_file = self.bbox_files[idx_video]
        dinamycImages = []
        frames_list = os.listdir(vid_name)
        
        frames_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
        if idx_next_segment < self.numFrames[idx_video]:
            segment, idx_next_segment = self.getOneSegment(frames_list, idx_next_segment)
            # print(segment)
            segment_info =  self.getSegmentInfo(segment, bbox_file)
            frames = []
            for frame in segment:
                img_dir = str(vid_name) + "/" + frame
                img1 = Image.open(img_dir).convert("RGB")
                img = np.array(img1)
                # print('Frame size: ', img.shape)
                frames.append(img)

            #### USING CPU NUMPY
            start_time = time.time()
            imgPIL, img = getDynamicImage(frames)
            end_time = time.time()
            # spend_time = end_time - start_time

            # ### USING CUDA PYTORCH
            # frames_tensor = torch.tensor(frames).type(torch.FloatTensor)
            # frames_tensor = frames_tensor.cuda()
            # torch.cuda.synchronize()
            # start_time = time.time()
            # imgPIL, img = computeDynamicImage(frames_tensor)
            # torch.cuda.synchronize()
            # end_time = time.time()
            spend_time = end_time - start_time
            # spend_time = 0
            
            imgPIL = self.spatial_transform(imgPIL.convert("RGB"))
            dinamycImages.append(imgPIL)
                
            dinamycImages = torch.stack(dinamycImages, dim=0)  #torch.Size([bs, ndi, ch, h, w])
            
            dinamycImages = dinamycImages.squeeze(dim=0)  ## get normal pytorch tensor [bs, ch, h, w]
            return dinamycImages, segment_info, idx_next_segment, spend_time
        else:
            return None, None, None, 0

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
    def getNumFrames(self, idx):
        return self.numFrames[idx]




