import numpy as np
from PIL import Image
import argparse
import torch
import os
import constants
import torchvision
import torchvision.transforms as transforms

def computeDynamicImage(frames):
    seqLen = frames.size()[0]
    if seqLen < 2:
      print('No se puede crear DI con solo un frames ...', seqLen)
    fw = np.zeros(seqLen)  
    for i in range(seqLen): #frame by frame
      fw[i] = np.sum(np.divide((2 * np.arange(i + 1, seqLen + 1) - seqLen - 1), np.arange(i + 1, seqLen + 1)))
    
    fw = torch.from_numpy(fw).float().cuda()
    sm = frames * fw[:,None, None, None]
    sm = sm.sum(0)
    sm = sm - torch.min(sm)
    sm = 255 * sm / torch.max(sm)
    img = sm
    img = img.cpu()
    imgPIL = Image.fromarray(np.uint8(img.numpy()))
    return imgPIL, img

def getCoeffiecient(index, seqLen):
  idx = np.sum(np.divide((2 * np.arange(index + 1, seqLen + 1) - seqLen - 1), np.arange(index + 1, seqLen + 1)))

def getDynamicImage(frames, savePath=None):
    seqLen = len(frames)
    if seqLen < 2:
      print('No se puede crear DI con solo un frames ...', seqLen)
    frames = np.stack(frames, axis=0)
    fw = np.zeros(seqLen)  
    for i in range(seqLen): #frame by frame
      fw[i] = np.sum(np.divide((2 * np.arange(i + 1, seqLen + 1) - seqLen - 1), np.arange(i + 1, seqLen + 1)))
    print('Di coeff=',fw)
    fwr = fw.reshape(seqLen, 1, 1, 1)  #coeficiebts
    sm = frames*fwr
    sm = sm.sum(0)
    sm = sm - np.min(sm)
    sm = 255 * sm / np.max(sm)
    img = sm.astype(np.uint8)
    ##to PIL image
    imgPIL = Image.fromarray(np.uint8(img))
    if savePath is not None:
      imgPIL.save(savePath)
    return imgPIL, img
