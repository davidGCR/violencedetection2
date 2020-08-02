from torch.utils.data import Dataset
from PIL import Image, ImageFile, ImageFilter
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
from UTIL.util import sortListByStrNumbers
import numpy as np

class RGBDataset(Dataset):
    def __init__(self, dataset,
                        labels,
                        numFrames,
                        frame_idx,
                        spatial_transform):
        self.videos = dataset
        self.labels = labels
        self.numFrames = numFrames
        self.frame_idx = frame_idx
        self.spatial_transform = spatial_transform
    
    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        vid_name = self.videos[idx]
        label = self.labels[idx]
        frames_names = os.listdir(vid_name)
        frames_names = sortListByStrNumbers(frames_names)
        frame_path = os.path.join(vid_name, frames_names[self.frame_idx])
        
        frame = Image.open(frame_path)
        frame = frame.convert("RGB")
        # frame = np.array(frame)
        frame = self.spatial_transform(frame)
        
        return vid_name, frame, label