from torch.utils.data import Dataset
from PIL import Image, ImageFile, ImageFilter
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import os
import cv2
import torch
import glob
import time
import VIDEO_REPRESENTATION.dynamicImage as dynamicImage
import random

class ViolenceDataset(Dataset):
    def __init__(self, dataset,
                        labels,
                        numFrames,
                        spatial_transform,
                        numDynamicImagesPerVideo,
                        videoSegmentLength,
                        positionSegment,
                        overlaping,
                        frame_skip,
                        skipInitialFrames,
                        preprocess_images):
        self.spatial_transform = spatial_transform
        self.videos = dataset
        self.labels = labels
        self.numFrames = numFrames
        self.numDynamicImagesPerVideo = numDynamicImagesPerVideo
        self.videoSegmentLength = videoSegmentLength
        self.positionSegment = positionSegment
        self.overlaping = overlaping
        self.frame_skip = frame_skip
        self.minSegmentLen = 5
        self.skipInitialFrames = skipInitialFrames
        self.preprocess_images = preprocess_images

    def __len__(self):
        return len(self.videos)
    
    def getindex(self, vid_name):
        matching = [s for s in self.videos if vid_name in s]
        
        if len(matching)>0:
            vid_name = matching[0]
            index = self.videos.index(vid_name)
            return index
        else:
            return None

    def getTemporalSegment(self, frames_list, start):
        segments_block = []
        if start == 0: #first segment
            block = []
            for i in range(start,start + self.videoSegmentLength):
                block.append(frames_list[i])
            # block = frames_list[start:start + self.videoBlockLength]
            idx_next_block = self.videoSegmentLength
            
        else:
            start = start - int(self.videoSegmentLength * self.overlaping)
            end = start + self.videoSegmentLength
            # block = frames_list[start:end]
            block = []
            for i in range(start,end):
                if i<len(frames_list):
                    block.append(frames_list[i])
            idx_next_block = end
            
        
        segments_block.append(block)
        # print('block: ',block)
        return block, start, idx_next_block, segments_block
        
    def getTemporalBlock(self, vid_name, idx_next_block):
        # vid_name = self.videos[idx_video]
        idx_video = self.getindex(vid_name)
        label = self.labels[idx_video]
        print('video buscado: {}, video encontrado: {}'.format(vid_name, self.videos[idx_video]))
        frames_list = os.listdir(vid_name)
        frames_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
        preprocessing_time = 0
        dinamycImages = []
        real_frames = None
        if idx_next_block < self.numFrames[idx_video]:
            block, start, idx_next_block, segments_block = self.getTemporalSegment(frames_list, idx_next_block)
            for seq in segments_block:
                # print('segment len:{}, segment: {}',len(seq), seq)
                frames = []
                real_frames = seq
                # r_frames = []
                for frame in seq:
                    img_dir = str(vid_name) + "/" + frame
                    img1 = Image.open(img_dir).convert("RGB")
                    img = np.array(img1)
                    frames.append(img)
                start_time = time.time()
                imgPIL, img = dynamicImage.getDynamicImage(frames)
                end_time = time.time()
                imgPIL = self.spatial_transform(imgPIL.convert("RGB"))
                
                preprocessing_time += (end_time - start_time)
                dinamycImages.append(imgPIL)
            # print('Len: ', len(dinamycImages), vid_name)
            dinamycImages = torch.stack(dinamycImages, dim=0)  #torch.Size([bs, ndi, ch, h, w])
            # if self.numDynamicImagesPerVideo == 1:
            #     dinamycImages = dinamycImages.squeeze(dim=0)  ## get normal pytorch tensor [bs, ch, h, w]
                
        return dinamycImages, label, idx_next_block, preprocessing_time, real_frames

    def getVideoSegmentsOverlapped(self, vid_name, idx):
        frames_list = os.listdir(vid_name)
        frames_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
        video_segments = []
        seqLen = self.videoSegmentLength
        o = int(self.overlaping * seqLen)
        indices = [x for x in range(0, self.numFrames[idx], self.frame_skip + 1)]
        indices_segments = [indices[x:x + seqLen] for x in range(0, len(indices), seqLen-o)]
        
        
        for indices_segment in indices_segments:
            segment = np.asarray(frames_list)[indices_segment].tolist()
            video_segments.append(segment)
            # video_segments.append(frames_list[indices_segment])
               
        return video_segments

    def checkSegmentLength(self, segment):
        return (len(segment) == self.videoSegmentLength or len(segment) > self.minSegmentLen)
    
    def padding(self, segment_list):
        if  len(segment_list) < self.numDynamicImagesPerVideo:
            last_element = segment_list[len(segment_list) - 1]
            for i in range(self.numDynamicImagesPerVideo - len(segment_list)):
                segment_list.append(last_element)
        elif len(segment_list) > self.numDynamicImagesPerVideo:
            segment_list = segment_list[0:self.numDynamicImagesPerVideo]
        return segment_list
    
    def segmentPreprocessing(self, vid_name, segment):
        frames = []
        for frame in segment:
            img_dir = str(vid_name) + "/" + frame
            if self.preprocess_images:
                img1 = Image.open(img_dir)
                img1 = img1.filter(ImageFilter.BoxBlur(5))
                img1 = img1.convert("RGB")
            else:   
                img1 = Image.open(img_dir).convert("RGB")
            img = np.array(img1)
            frames.append(img)
        return frames
           
    def getVideoSegments(self, vid_name, idx):
        frames_list = os.listdir(vid_name)
        frames_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
        if self.skipInitialFrames < len(frames_list):
            le = len(frames_list)
            frames_list = frames_list[self.skipInitialFrames:le]
            self.numFrames[idx] -= self.skipInitialFrames
        video_segments = []
        
        indices = [x for x in range(0, self.numFrames[idx], self.frame_skip + 1)]
        seqLen = self.videoSegmentLength
        overlap_length = int(self.overlaping*seqLen)
        indices_segments = [indices[x:x + seqLen] for x in range(0, len(indices), seqLen-overlap_length)]
        # print('indices1: ', indices_segments)
        indices_segments_cpy = []
        for i,seg in enumerate(indices_segments): #Verify is a segment has at least 2 or more frames.
            if self.checkSegmentLength(seg):
                indices_segments_cpy.append(indices_segments[i])
        indices_segments = indices_segments_cpy
        # print('indices2: ',indices_segments)
        indices_segments = self.padding(indices_segments) #If numDynamicImages < wanted the padding else delete
        # print('indices3: ', len(indices_segments), indices_segments)

        for i, indices_segment in enumerate(indices_segments): #Generate segments using indices
            segment = np.asarray(frames_list)[indices_segment].tolist()
            video_segments.append(segment)
                
        return video_segments

    def __getitem__(self, idx):
        vid_name = self.videos[idx]
        # print(vid_name)
        label = self.labels[idx]
        dinamycImages = []
        video_segments = self.getVideoSegments(vid_name, idx) # bbox_segments: (1, 16, 6)= (no segments,no frames segment,info
        preprocessing_time = 0.0
        for seq in video_segments:
            frames = self.segmentPreprocessing(vid_name, seq)
            start_time = time.time()
            imgPIL, img = dynamicImage.getDynamicImage(frames)
            end_time = time.time()
            imgPIL = self.spatial_transform(imgPIL.convert("RGB"))
            
            preprocessing_time += (end_time - start_time)
            dinamycImages.append(imgPIL)
        # print('Len: ', len(dinamycImages), vid_name)
        dinamycImages = torch.stack(dinamycImages, dim=0)  #torch.Size([bs, ndi, ch, h, w])
        # print(dinamycImages.size())
        # if self.numDynamicImagesPerVideo == 1:
        #     dinamycImages = dinamycImages.squeeze(dim=0) ## get normal pytorch tensor [bs, ch, h, w]
        return dinamycImages, label, vid_name, preprocessing_time #dinamycImages, label:  <class 'torch.Tensor'> <class 'int'> torch.Size([3, 224, 224])




