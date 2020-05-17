from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
# import cv2
import torch
import glob
import time
import VIDEO_REPRESENTATION.dynamicImage as dynamicImage
import random
import sys
import torchvision.transforms as transforms
from torch.utils.data._utils.collate import default_collate

class ViolenceOnlineDataset(Dataset):
    # dataset, labels, numFrames, spatial_transform, numDynamicImagesPerVideo, videoSegmentLength, positionSegment
    def __init__(self, dataset, labels, numFrames, spatial_transform, nDynamicImages, overlapping):
        self.spatial_transform = spatial_transform
        self.videos = dataset
        self.labels = labels
        self.numFrames = numFrames  #dataset total num frames by video
        self.overlapping = overlapping

    def __len__(self):
        return len(self.videos)

    def getSegment(self, frames_list, start):
        if start == 0: #first segment
            segment = frames_list[start:start + self.videoSegmentLength]
            idx_next_segment = self.videoSegmentLength
        else:
            start = start - int(self.videoSegmentLength * self.overlapping)
            end = start + self.videoSegmentLength
            segment = frames_list[start:end]
            idx_next_segment = end
        
        return segment, idx_next_segment
    
    def computeSegmentDynamicImg(self, idx_video, idx_next_segment):
        vid_name = self.videos[idx_video]
        bbox_file = self.bbox_files[idx_video]
        dinamycImages = []
        frames_list = os.listdir(vid_name)
        frames_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
        if idx_next_segment < self.numFrames[idx_video]:
            segment, idx_next_segment = self.getSegment(frames_list, idx_next_segment)
            frames = []
            for frame in segment:
                img_dir = str(vid_name) + "/" + frame
                img1 = Image.open(img_dir).convert("RGB")
                img = np.array(img1)
                frames.append(img)

            imgPIL, img = dynamicImage.getDynamicImage(frames)
            imgPIL = self.spatial_transform(imgPIL.convert("RGB"))
            dinamycImages.append(imgPIL)
                
            dinamycImages = torch.stack(dinamycImages, dim=0)  #torch.Size([bs, ndi, ch, h, w])
            if self.numDynamicImagesPerVideo == 1:
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
            # print('ONLY video: ',matching,index)
            return index
        else:
            return None
        # # print(vid_name)
        # label = self.labels[idx]
        
        # return vid_name, label




