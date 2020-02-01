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

class AnomalyDatasetAumented(Dataset):
    def __init__(self, images, labels, spatial_transform):
        self.spatial_transform = spatial_transform
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):
        img_name = self.images[idx]
        # print(vid_name)
        label = self.labels[idx]
        img1 = Image.open(img_name).convert("RGB")
        dynamicImage = self.spatial_transform(img1)
        return dynamicImage, label, 0, 0

    



