import numpy as np
from PIL import Image
import argparse
import torch
import os
import constants
import torchvision
import torchvision.transforms as transforms
import util
# class DinamycImage():
#     def __init__(self,):

def getDynamicImage(frames):
    seqLen = len(frames)
    if seqLen < 2:
      print('No se puede crear DI con solo un frames ...', seqLen)
    
    frames = np.stack(frames, axis=0)

    fw = np.zeros(seqLen)  
    for i in range(seqLen): #frame by frame
      fw[i] = np.sum( np.divide((2*np.arange(i+1,seqLen+1)-seqLen-1) , np.arange(i+1,seqLen+1))  )

    fwr = fw.reshape(seqLen,1,1,1)
    sm = frames*fwr
    sm = sm.sum(0)
    sm = sm - np.min(sm)
    # print('*************** np.max(sm) : ', str(np.max(sm)))
    sm = 255 * sm /np.max(sm) 
    img = sm.astype(np.uint8)
    ##to PIL image
    imgPIL = Image.fromarray(np.uint8(img))
    return imgPIL, img

def getSequences(vid_name, numDI, sequenceLength):
  frames_list = os.listdir(vid_name)
  frames_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
  seqLen = 0
  if sequenceLength == 0:
    num_frames_on_video = len(frames_list)
  else:
    num_frames_on_video = sequenceLength if len(frames_list) >= sequenceLength else len(frames_list)

  # num_frames_on_video = len(frames_list)
  print('num_frames_on_video: ', num_frames_on_video)
  
  seqLen = num_frames_on_video // numDI
  sequences = [frames_list[x:x + seqLen] for x in range(0, num_frames_on_video, seqLen)]
  # sequences = []
  # for x in range(0, num_frames_on_video, seqLen):
  #     # print('x: ', x, x+seqLen, len(frames_list))
  #     sequences.append(frames_list[x:x + seqLen])

  if len(sequences) > numDI:
      diff = len(sequences) - numDI
      sequences = sequences[: - diff]
  # if len(sequences) < self.nDynamicImages:
  #     print('-->len(sequences)',len(sequences))
  return sequences

def computeDynamicImages(videoPath, numDynamicImages, sequenceLength=0):
  print('*'*30, 'Imagen dinamica original')
  sequences = getSequences(videoPath, numDynamicImages,sequenceLength)
  
  dynamicImages = []

  transform = transforms.Compose([
    # transforms.RandomResizedCrop(224),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.49237782, 0.49160805, 0.48998737], [0.11053326, 0.11088469, 0.11275752])
  ])

  for seq in sequences:
    # if len(seq) == seqLen:
    frames = []
    for frame in seq:
        img_dir = os.path.join(videoPath,frame)
        # print(img_dir)
        img = Image.open(img_dir).convert("RGB")
        img = np.array(img)
        frames.append(img)
    imgPIL, img = getDynamicImage(frames)
    imgPIL = transform(imgPIL)
    dynamicImages.append(imgPIL)               
  
  dinamycImages = torch.stack(dynamicImages, dim=0)

  # images = torchvision.utils.make_grid(dinamycImages.cpu().data, padding=10)
  return dinamycImages

  
# def __main__():
#   parser = argparse.ArgumentParser()
#   parser.add_argument("--framesPath", type=str)
#   parser.add_argument("--numDynamicImages", type=int)
#   args = parser.parse_args()
#   videoPath = args.framesPath
#   videoPath = os.path.join(constants.PATH_UCFCRIME2LOCAL_FRAMES_REDUCED,videoPath)
#   numDynamicImages = args.numDynamicImages
#   sequences = getSequences(videoPath, numDynamicImages)
#   print(len(sequences))
#   dynamicImages = []

#   transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize([0.4770381, 0.4767955, 0.4773611], [0.11147115, 0.11427314, 0.11617025])
#   ])

#   for seq in sequences:
#     # if len(seq) == seqLen:
#     frames = []
#     for frame in seq:
#         img_dir = os.path.join(videoPath,frame)
#         # print(img_dir)
#         img = Image.open(img_dir).convert("RGB")
#         img = np.array(img)
#         frames.append(img)
#     imgPIL, img = getDynamicImage(frames)
#     imgPIL = transform(imgPIL)
#     dynamicImages.append(imgPIL)               
  
#   dinamycImages = torch.stack(dynamicImages, dim=0)

#   images = torchvision.utils.make_grid(dinamycImages.cpu().data, padding=10)
#   util.imshow(images,'')

    # imgPIL = self.spatial_transform(imgPIL.convert("RGB"))


# __main__()