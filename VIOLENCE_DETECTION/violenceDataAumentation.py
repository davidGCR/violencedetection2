
import os
from dynamicImage import *
from PIL import Image
import numpy as np
import initializeDataset
import constants

class DataAumentation():
    def __init__(self, dataset, labels, numFrames, numDynamicImagesPerVideo, segmentLength, overlaping, output_path_violence, output_path_nonviolence):
        self.videos = dataset
        self.labels = labels
        self.numFrames = numFrames
        self.numDynamicImagesPerVideo = numDynamicImagesPerVideo
        self.output_path_violence = output_path_violence
        self.output_path_nonviolence = output_path_nonviolence
        self.overlaping = overlaping
        self.segmentLength = segmentLength
        # self.videoSegmentLength = videoSegmentLength

    def __len__(self):
        return len(self.videos)

    def getVideoSegments(self, vid_name, idx):
        frames_list = os.listdir(vid_name)
        frames_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
        video_segments = []
        if self.numDynamicImagesPerVideo is not None:
            seqLen = int(len(frames_list) / self.numDynamicImagesPerVideo)
            num_frames_on_video = seqLen*self.numDynamicImagesPerVideo 
            video_segments = [frames_list[x:x + seqLen] for x in range(0, num_frames_on_video, seqLen)]
        else:
            seqLen = self.segmentLength
            num_frames_overlapped = int(self.overlaping * seqLen)
            video_segments.append(frames_list[0:seqLen])
            i = seqLen #20
            end = 0
            while i<len(frames_list):
                start = i - num_frames_overlapped #10 20 30
                end = start + seqLen  #30 40 50
                i = end #30 40
                if end < len(frames_list):
                    video_segments.append(frames_list[start:end])
        # num_frames_on_video = self.numFrames[idx]
        
        return video_segments

    def __getitem__(self, idx):
        vid_name = self.videos[idx]
        label = self.labels[idx]
        video_segments = self.getVideoSegments(vid_name, idx) # bbox_segments: (1, 16, 6)= (no segments,no frames segment,info
        for idx, seq in enumerate(video_segments):
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
            # imgPIL = self.spatial_transform(imgPIL.convert("RGB"))
            head, tail = os.path.split(vid_name)
            # idx = tail+'_'+str(idx+1)
            if label == 0:
                folder_out = os.path.join(self.output_path_nonviolence,tail)
            elif label == 1:
                folder_out = os.path.join(self.output_path_violence, tail)
            
            if not os.path.exists(folder_out):
                os.makedirs(folder_out)
            imgPIL.save(os.path.join(folder_out,str(idx)+'.png'))

            # dinamycImages.append(imgPIL)
        return vid_name

def main():
    hockey_path_violence = constants.PATH_HOCKEY_FRAMES_VIOLENCE
    hockey_path_noviolence = constants.PATH_HOCKEY_FRAMES_NON_VIOLENCE
    output_path_violence = constants.PATH_HOCKEY_AUMENTED_VIOLENCE
    output_path_nonviolence = constants.PATH_HOCKEY_AUMENTED_NON_VIOLENCE
    shuffle = False

    datasetAll, labelsAll, numFramesAll = initializeDataset.createDataset(hockey_path_violence, hockey_path_noviolence, shuffle)  #shuffle
    numDynamicImagesPerVideo = None
    overlaping = 0.5
    segmentLength = 20

    dA = DataAumentation(datasetAll, labelsAll, numFramesAll, numDynamicImagesPerVideo,segmentLength, overlaping, output_path_violence, output_path_nonviolence)
    for video in dA:
        print(video)


if __name__ == '__main__':
    main()