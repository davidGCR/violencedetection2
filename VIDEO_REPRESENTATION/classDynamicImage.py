import numpy as np
from PIL import Image
import argparse
import torch
import os
import constants
import torchvision
import torchvision.transforms as transforms
import util
import initializeDataset
from FPS import FPSMeter
import cv2
import time

class DynamicImage():
  def __init__(self, dataset, labels, numFrames, numDynamicImagesPerVideo, videoSegmentLength, overlaping):
    self.videos = dataset
    self.labels = labels
    self.numFrames = numFrames
    self.videoSegmentLength = videoSegmentLength
    self.overlaping = overlaping
    self.numDynamicImagesPerVideo = numDynamicImagesPerVideo
  
  def getVideoSegmentsOverlapped(self, vid_name, idx):
    frames_list = os.listdir(vid_name)
    frames_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    video_segments = []
    seqLen = self.videoSegmentLength
    num_frames_overlapped = int(self.overlaping * seqLen)
    # print('num_frames_overlapped: ', num_frames_overlapped, len(frames_list), self.numFrames[idx])
    video_segments.append(frames_list[0:seqLen])
    i = seqLen -1#20
    end = 0
    while i<len(frames_list):
        if len(video_segments) == self.numDynamicImagesPerVideo:
            break
        else:
          start = i - num_frames_overlapped + 1 #10 20 
          end = start + seqLen-1  #29 39
          i = end #29 39
          if end < len(frames_list):
              video_segments.append(frames_list[start:end])
          elif len(frames_list) - start > 3:
              end = len(frames_list)-1
              video_segments.append(frames_list[start:end])
            # break
        # if len(video_segments) == self.numDynamicImagesPerVideo:
        #     break 
    return video_segments

  def summarizeVideo(self, idx):
    vid_name = self.videos[idx]
    label = self.labels[idx]
    sequences = self.getVideoSegmentsOverlapped(vid_name, idx)
    # print(len(sequences))
    # print(sequences)
    dinamycImages = []
    for seq in sequences:
      # print(len(seq))
      frames = []
      # r_frames = []
      for frame in seq:
          img_dir = str(vid_name) + "/" + frame
          img1 = Image.open(img_dir).convert("RGB")
          img = np.array(img1)
          frames.append(img)
      start_time = time.time()
      imgPIL, img = self.getDynamicImage(frames)
      end_time = time.time()
      spend_time  = end_time-start_time
      dinamycImages.append(img)
    return vid_name, label, dinamycImages, spend_time

  def computeDynamicImage(self,frames):
      # print('compute DY: ', type(frames[0]), frames.size()) #torch.Size([30, 240, 320, 3])
      seqLen = frames.size()[0]

      if seqLen < 2:
        print('No se puede crear DI con solo un frames ...', seqLen)

      fw = np.zeros(seqLen)  
      for i in range(seqLen): #frame by frame
        fw[i] = np.sum(np.divide((2 * np.arange(i + 1, seqLen + 1) - seqLen - 1), np.arange(i + 1, seqLen + 1)))
      
      fw = torch.from_numpy(fw).float().cuda()
      
      sm = frames * fw[:,None, None, None]
      sm = sm.sum(0)
      # print('min :', torch.min(sm))

      sm = sm - torch.min(sm)
      # print('*************** np.max(sm) : ', str(np.max(sm)))
      sm = 255 * sm / torch.max(sm)
      # print('max :', torch.max(sm))
      img = sm
      #img = sm.type(dtype=torch.uint8)
      img = img.cpu()
      imgPIL = Image.fromarray(np.uint8(img.numpy()))
      return imgPIL, img

  def getDynamicImage(self, frames):
      seqLen = len(frames)

      if seqLen < 2:
        print('No se puede crear DI con solo un frames ...', seqLen)
      
      frames = np.stack(frames, axis=0) #frames:  (30, 240, 320, 3)
      # print('frames: ', frames.shape)

      fw = np.zeros(seqLen)  
      for i in range(seqLen): #frame by frame
        fw[i] = np.sum( np.divide((2*np.arange(i+1,seqLen+1)-seqLen-1) , np.arange(i+1,seqLen+1))  )
      # print(fw)
      fwr = fw.reshape(seqLen, 1, 1, 1)  #coeficiebts
      # print('fwr0000: ', fwr.shape)
      #print('Frames: ',frames.shape, 'Di coeff: ', fwr.shape)
      sm = frames*fwr
      sm = sm.sum(0)
      # print('min :', np.min(sm))
      sm = sm - np.min(sm)
      # print('*************** np.max(sm) : ', str(np.max(sm)))
      sm = 255 * sm / np.max(sm)
      # img = sm
      # print('max :', np.max(sm))
      img = sm.astype(np.uint8)
      # print('IMG final : min :', np.min(img), 'max:',np.max(img))
      # print('IMG00000: ',img.shape, )
      ##to PIL image
      imgPIL = Image.fromarray(np.uint8(img))
      return imgPIL, img
  
def __main__():
  parser = argparse.ArgumentParser()
  parser.add_argument("--videoSegmentLength", type=int)
  parser.add_argument("--overlaping", type=float)
  parser.add_argument("--numDynamicImagesPerVideo", type=int)
  parser.add_argument("--plot", type=lambda x: (str(x).lower() == 'true'), default=False)
  args = parser.parse_args()
  videoSegmentLength = args.videoSegmentLength
  overlaping = args.overlaping
  numDynamicImagesPerVideo = args.numDynamicImagesPerVideo
  plot = args.plot
  shuffle = False
  datasetAll, labelsAll, numFramesAll = initializeDataset.createDataset(constants.PATH_HOCKEY_FRAMES_VIOLENCE, constants.PATH_HOCKEY_FRAMES_NON_VIOLENCE, shuffle)  #shuffle
  dynamicImage = DynamicImage(datasetAll, labelsAll, numFramesAll, numDynamicImagesPerVideo, videoSegmentLength, overlaping)

  fpsMeter = FPSMeter()

  for i, v in enumerate(datasetAll):
    
    # start_time = time.time()
    vid_name, label, dinamycImages, ttime = dynamicImage.summarizeVideo(i)
    # end_time = time.time()
    # ttime = end_time - start_time
    fpsMeter.update(ttime)
    print(vid_name)
    if plot:
      s = dinamycImages[0].shape
      # print(s)
      BORDER = np.ones((s[0],20,3))
      if len(dinamycImages)>1:
        img = np.concatenate(dinamycImages, axis=1)
      else:
        img = dinamycImages[0]
      cv2.imshow(vid_name, img)
      if cv2.waitKey(30) & 0xFF == ord('q'):
            break 
  fpsMeter.print_statistics()
    # 


__main__()