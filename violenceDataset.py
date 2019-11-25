from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import cv2
import torch
import glob
import time
from dynamicImage import *


class ViolenceDataset(Dataset):
    def __init__(self, dataset, labels, spatial_transform, source, interval_duration=0.0, difference=3, maxDuration=0, nDynamicImages=0, debugg_mode = False):
        self.spatial_transform = spatial_transform
        self.images = dataset
        self.labels = labels
        self.interval_duration = interval_duration
        self.diference_max = difference
        self.nDynamicImages = nDynamicImages
        self.maxDuration = maxDuration
        self.source = source
        self.debugg_mode = debugg_mode

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        label = self.labels[idx]
        video_name = self.images[idx]
        dinamycImages = self.fromFrames(idx)
        dinamycImages = torch.stack(dinamycImages, 0)
        if self.nDynamicImages == 1:
            dinamycImages = dinamycImages.squeeze(dim=0)
        # print('dinamycImages, label: ',type(dinamycImages), dinamycImages.size(), type(label), label)
        return dinamycImages, label, video_name #dinamycImages, label:  <class 'torch.Tensor'> <class 'int'> torch.Size([3, 224, 224])

    def fromFrames(self, idx):
        vid_name = self.images[idx]
        dinamycImages = []
        frames_list = os.listdir(vid_name)
        frames_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
        total_frames = len(frames_list)
        seqLen = int(total_frames/self.nDynamicImages)
        sequences = [
            frames_list[x : x + seqLen] for x in range(0, total_frames, seqLen)
        ]
        # print('nDynamicImages: ', self.nDynamicImages)
        # print('total_frames: ', total_frames)
        # print('seqLen: ', seqLen)
        # print('Num sequences: ', len(sequences))
        for seq in sequences:
            if len(seq) == seqLen:
                frames = []
                for frame in seq:
                    img_dir = str(vid_name) + "/" + frame
                    img = Image.open(img_dir).convert("RGB")
                    img = np.array(img)
                    frames.append(img)
                imgPIL, img = getDynamicImage(frames)
                dinamycImages.append(self.spatial_transform(imgPIL.convert("RGB")))
        return dinamycImages

    def fromVideo(self, idx):
        vid_name = self.images[idx]
        dinamycImages = []
        cap = cv2.VideoCapture(vid_name)
        # start_time_ms = time.time()
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # fps = cap.get(cv2.CAP_PROP_FPS)
        # duration = video_length / fps
        # print('video durationnnnnnnnnnnnnnnnnnn: ')
        count = 0
        success = True
        frames = []
        # self.nDynamicImages = int(
        #     duration / self.interval_duration
        # )  # Number of Dynamic images estimated
        numberFramesInterval = 0
        capture_duration = self.interval_duration
        while count < video_length - 1:
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            # print('current_time ',current_time, 'capture_duration ',capture_duration) 
            if current_time <= capture_duration:
                success, image = cap.read()
                if success:
                    frames.append(image)
            else:
                numberFramesInterval = len(frames)  ##number of frames to sumarize
                # print('to summarize: ', numberFramesInterval)
                img = getDynamicImage(frames)

                dinamycImages.append(self.spatial_transform(img.convert("RGB")))
                frames = []
                frames.append(image)
                capture_duration = current_time + self.interval_duration
            count += 1
        ## the last chunk
        if numberFramesInterval - len(frames) < self.diference_max: 
            # print('ading the last ----------: ')
            imgPIL, img = getDynamicImage(frames)
            dinamycImages.append(self.spatial_transform(imgPIL.convert("RGB")))  ##add dynamic image
        if len(dinamycImages) > self.nDynamicImages:
            n = len(dinamycImages) - self.nDynamicImages
            del dinamycImages[-n:]



