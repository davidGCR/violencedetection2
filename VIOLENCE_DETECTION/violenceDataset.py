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
from VIDEO_REPRESENTATION.preprocessor import Preprocessor
from VIDEO_REPRESENTATION.frameExtractor import FrameExtractor
import random
import torchvision.transforms as transforms
from operator import itemgetter
from UTIL.bbox_gt_loader import load_bbox_gt
from PREPROCESING.segmentation import bbox_from_di
# from PREPROCESING.segmentation import denoise, cluster_segmentation

class ViolenceDataset(Dataset):
    def __init__(self, videos,
                        labels,
                        numFrames,
                        spatial_transform,
                        numDynamicImagesPerVideo,
                        videoSegmentLength,
                        positionSegment='begin',
                        overlaping=0,
                        frame_skip=0,
                        skipInitialFrames=0,
                        ppType=None,
                        useKeyframes=None,
                        windowLen=None,
                        dataset=None):
        if spatial_transform is None:
            self.spatial_transform = transforms.Compose([
                # transforms.RandomResizedCrop(224),
                # transforms.CenterCrop(224),
                transforms.ToTensor()])
        else:
            self.spatial_transform = spatial_transform
        self.videos = videos
        self.labels = labels
        self.numFrames = numFrames
        self.numDynamicImagesPerVideo = numDynamicImagesPerVideo
        self.videoSegmentLength = videoSegmentLength
        self.positionSegment = positionSegment
        self.overlaping = overlaping
        self.frame_skip = frame_skip
        self.minSegmentLen = 5
        self.skipInitialFrames = skipInitialFrames
        self.ppType = ppType  #preprocessing
        self.preprocessor = Preprocessor(ppType)
        self.useKeyframes = useKeyframes
        self.extractor = FrameExtractor(len_window=windowLen)
        self.dataset = dataset

    def __len__(self):
        return len(self.videos)

    def setTransform(self, transform):
        self.spatial_transform = transform

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

    def loadFramesSeq(self, vid_name, sequence):
        frames = []
        frames_paths = []
        for frame in sequence:
            img_dir = str(vid_name) + "/" + frame
            frames_paths.append(img_dir)
            img1 = Image.open(img_dir)
            img1 = img1.convert("RGB")
            img = np.array(img1)
            frames.append(img)
        return frames, frames_paths

    def getVideoSegments(self, vid_name, idx):
        frames_list = os.listdir(vid_name)
        frames_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
        # print(frames_list)
        if self.skipInitialFrames > 0 and self.skipInitialFrames < len(frames_list) and self.numFrames[idx]>40:
            le = len(frames_list)
            frames_list = frames_list[self.skipInitialFrames:le]
            self.numFrames[idx] -= self.skipInitialFrames
        video_segments = []
        # print('self.numFrames = ', type(self.numFrames))
        indices = [x for x in range(0, self.numFrames[idx], self.frame_skip + 1)]

        if self.videoSegmentLength == 0:
            seqLen = self.numFrames[idx]
        else:
            seqLen = self.videoSegmentLength
        # print('seqLen = ', seqLen)
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

        return video_segments, indices_segments

    def __getitem_diff__(self, vid_name):
        dynamicImages = []
        sequence = os.listdir(vid_name)
        sequence.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
        video_segment, paths = self.loadFramesSeq(vid_name, sequence)
        candidate_frames, frames_indexes = self.extractor.__extract_candidate_frames_fromFramesList__(video_segment)
        # print('Frames indexes-{}/{}='.format(len(sequence),len(frames_indexes)))
        imgPIL, img = dynamicImage.getDynamicImage(candidate_frames)
        imgPIL = self.spatial_transform(imgPIL.convert("RGB"))
        dynamicImages.append(imgPIL)
        return dynamicImages

    def __getitem_blur__(self, vid_name):
        dynamicImages = []
        nelem = 1
        sequence = os.listdir(vid_name)
        sequence.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
        frames, paths = self.loadFramesSeq(vid_name, sequence)

        _, indices_segments = self.getVideoSegments(vid_name, idx)
        # print(len(frames),len(indices_segments))
        # ca2@i@VnTCC5TDK
        segment_idxs = []
        for i,idxs in enumerate(indices_segments):
            frames_in_segment = list(itemgetter(*idxs)(frames)) #np.asarray(frames)[idxs].tolist()
            blurrings = self.extractor.__compute_frames_blurring_fromList__(frames_in_segment, plot=False)
            indexes_candidates = self.extractor.__candidate_frames_blur_based__(frames_in_segment, blurrings, self.useKeyframes, nelem=nelem)

            if nelem > 1:
                indexes_candidates = list(itemgetter(*indexes_candidates)(idxs))
                segment_idxs += indexes_candidates
            else:
                indexes_candidates = idxs[indexes_candidates[0]]
                segment_idxs.append(indexes_candidates)
        segment_idxs.sort()
        candidate_frames = list(itemgetter(*segment_idxs)(frames))

        imgPIL, img = dynamicImage.getDynamicImage(candidate_frames)
        imgPIL = self.spatial_transform(imgPIL.convert("RGB"))
        dynamicImages.append(imgPIL)
        return dynamicImages

    def __getitem__(self, idx):
        vid_name = self.videos[idx]
        # print(vid_name, self.numFrames[idx])
        label = self.labels[idx]
        dynamicImages = []
        ipts = []
        # preprocessing_time = 0.0
        video_segments, _ = self.getVideoSegments(vid_name, idx)  # bbox_segments: (1, 16, 6)= (no segments,no frames segment,info
        paths = []
        for i, sequence in enumerate(video_segments):
            video_segments[i], pths = self.loadFramesSeq(vid_name, sequence)
            paths.append(pths)
        for sequence in video_segments:
            # start_time = time.time()
            imgPIL, img = dynamicImage.getDynamicImage(sequence)

            # end_time = time.time()
            # preprocessing_time += (end_time - start_time)
            dynamicImages.append(np.array(imgPIL))
            ipts.append(self.spatial_transform(imgPIL.convert("RGB")))
        ipts = torch.stack(ipts, dim=0)  #torch.Size([bs, ndi, ch, h, w])
        m_bboxes = bbox_from_di(dynamicImages, num_imgs=5, plot=False)

        gt_bboxes, one_box = None, None
        if self.dataset == 'ucfcrime2local':
            gt_bboxes, one_box = load_bbox_gt(vid_name, label, paths[0])
        else:
            one_box = m_bboxes
            print('m_bboxes: ', len(one_box))
            # gt_bboxes, one_box = [-1, -1, -1, -1], [0, 0, 224, 224]

        one_box=torch.from_numpy(np.array(one_box)).float()

        return (ipts, idx, dynamicImages, one_box), label
        # return ipts, dynamicImages, label, vid_name, one_box, paths #dinamycImages, label:  <class 'torch.Tensor'> <class 'int'> torch.Size([3, 224, 224])

# if __name__ == '__main__':
