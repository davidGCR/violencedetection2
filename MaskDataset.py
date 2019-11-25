from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import cv2
import torch
import glob
import time
from dynamicImage import *


class MaskDataset(Dataset):
    def __init__(self, dataset, labels, spatial_transform, source, difference, maxDuration, nDynamicImages, saliency_model):
        self.spatial_transform = spatial_transform
        self.images = dataset
        self.labels = labels
        self.diference_max = difference
        self.nDynamicImages = nDynamicImages
        self.maxDuration = maxDuration
        self.source = source
        self.saliency_model = saliency_model
        # self.saliency_model.cuda()
        self.saliency_model.eval()
        input_size = 224
        self.ones = torch.ones(1, input_size, input_size)
        self.zeros = torch.zeros(1, input_size, input_size)

        # self.ones = self.ones.cuda()
        # self.zeros = self.zeros.cuda()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        label = self.labels[idx]
        label_t = torch.tensor([label])
        dinamycImages = self.fromFrames(idx)
        dinamycImages = torch.stack(dinamycImages, 0)
        # dinamycImages.cuda()
        # print('dinamycImages: ', dinamycImages.size())
        # if self.nDynamicImages == 1:
        # dinamycImages = dinamycImages.squeeze(dim=0)
        # print('dinamycImages, labels: ',type(dinamycImages),dinamycImages.size(), type(label_t), label_t.size(), label)
        masks, _ = self.saliency_model(dinamycImages, label_t)
        y = torch.where(masks > masks.view(masks.size(0), masks.size(1), -1).mean(2)[:,:, None, None], self.ones, self.zeros)
        maskedImages = dinamycImages * y
        maskedImages = maskedImages.squeeze(dim=0)
        # print('maskedImages, labels: ',type(maskedImages),maskedImages.size(), type(label_t), label_t.size(), label)
        return maskedImages, label

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


