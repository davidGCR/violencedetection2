import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from PIL import Image, ImageFile
import matplotlib.pyplot as plt
from UTIL.util import sortListByStrNumbers
import math
import cv2
import constants

class FrameSampler():
    def __init__(self, minSampleLength, threshold1, threshold2):
        self.minSampleLength = minSampleLength
        self.threshold1 = threshold1
        self.threshold2 = threshold2

    def grayCentroid(self, numpyImg):
        [h, w] = numpyImg.shape #(288, 360)
        x_c = 0
        for i in range(h):
            for j in range(w):
                x_c += numpyImg[i,j]*(j+1)
        x_c = x_c/np.sum(numpyImg)
        y_c = 0
        for i in range(h):
            for j in range(w):
                y_c += numpyImg[i,j]*(i+1)
        y_c = y_c/np.sum(numpyImg)
        return x_c, y_c

    def centroidDistance(self, x_c1, y_c1, x_c2, y_c2):
        l = ((x_c1-x_c2)**2) + ((y_c1-y_c2)**2) 
        return l

    def relativeDistance(self, x_c1, y_c1, x_c2, y_c2):
        d = self.centroidDistance(x_c1, y_c1, x_c2, y_c2) / math.sqrt(x_c2 ** 2 + y_c2 ** 2)
        return d
    
    def averageCentroid(self, similar_frames_c):
        x_c = [pair[0] for pair in similar_frames_c]
        y_c = [pair[1] for pair in similar_frames_c]
        x_c = np.array(x_c)
        y_c = np.array(y_c)
        x_c_start = np.average(x_c)
        y_c_start = np.average(y_c)
        return x_c_start, y_c_start

    
    def plotFrame(self, npImg0, i0, npImg1, i1):
        npImg0 = np.stack((npImg0,) * 3, axis=-1)
        npImg1 = np.stack((npImg1,) * 3, axis=-1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(npImg0, str(i0), (0, 25), font, 0.6, constants.yellow, 2, cv2.LINE_AA)
        cv2.putText(npImg1, str(i1), (0, 25), font, 0.6, constants.yellow, 2, cv2.LINE_AA)
        
        cv2.imshow('frame0', npImg0)
        cv2.imshow('frame1', npImg1)
        pos_x = 20
        sep = 400
        cv2.namedWindow("frame0");#x,y
        cv2.moveWindow("frame0", pos_x, 100);

        cv2.namedWindow("frame1");
        cv2.moveWindow("frame1", pos_x+sep, 100);
        
        if cv2.waitKey(50) & 0xFF == ord('q'):
            return
    
    def sample(self, video_path):
        frames_dirs = os.listdir(video_path)
        frames_dirs = sortListByStrNumbers(frames_dirs)
        frameT0 = Image.open(os.path.join(video_path, frames_dirs[0])).convert('L')
        frameT0 = np.array(frameT0)
        # self.plotFrame(frameT0)
        frameT1 = 0
        x_c0, y_c0 = self.grayCentroid(frameT0)
        frames_centroids = [[x_c0, y_c0]]
        
        frames_samples = [frameT0]
        frames_samples_indices = [0]

        d_v = 0
        for i in range(1, len(frames_dirs)):
            frameT1 = Image.open(os.path.join(video_path,frames_dirs[i])).convert('L')
            frameT1 = np.array(frameT1)
    
            self.plotFrame(frameT0, i - 1, frameT1, i)
            
            [x_c0, y_c0] = frames_centroids[len(frames_centroids)-1]
            x_c1, y_c1 = self.grayCentroid(frameT1)
            
            # if len(frames_centroids) < self.minSampleLength:
            x_c_start, y_c_start = self.averageCentroid(frames_centroids)
            d = self.relativeDistance(x_c_start, y_c_start, x_c1, y_c1)

            if d < self.threshold1:
                frames_centroids.append([x_c1, y_c1])
            else:
                frames_samples.append(frameT1)
                frames_samples_indices.append(i)
            print('frame t({})-frame t+1({}) = {}'.format(i-1,i, d))
            frameT0 = frameT1
        return frames_samples, frames_samples_indices

def main():
    # video_dir = '/Users/davidchoqueluqueroman/Desktop/PAPERS-CODIGOS/violencedetection2/DATASETS/HockeyFightsDATASET/frames/violence/1'
    video_dir = '/Users/davidchoqueluqueroman/Desktop/PAPERS-CODIGOS/violencedetection2/DATASETS/CrimeViolence2LocalDATASET/frames/nonviolence/Normal_Videos-Arrest002-NSplit-2'
    Sampler = FrameSampler(minSampleLength=4, threshold1=0.002, threshold2=0.3)
    frames_samples, frames_samples_indices = Sampler.sample(video_dir)
    print('LenSampled: ', len(frames_samples))

    for i,frame in enumerate(frames_samples):
        cv2.imshow(str(frames_samples_indices[i]), frame)
        while cv2.waitKey(50) & 0xFF == ord('q'):
            break

    # image_dir = '/Users/davidchoqueluqueroman/Desktop/PAPERS-CODIGOS/violencedetection2/DATASETS/HockeyFightsDATASET/frames/nonviolence/73/frame003.jpg'
    # frames_dirs = os.listdir(video_dir)
    # frames_dirs = sortListByStrNumbers(frames_dirs)

    # for frame in frames_dirs:
    #     image = Image.open(os.path.join(video_dir,frame)).convert('L')
    #     image = np.array(image)
    #     # print('image shape: ', image.shape)
    #     plt.imshow(image)
    #     x_c, y_c = grayCentroid(image)
    #     print('x_c, y_c: ', x_c, y_c)
    

if __name__ == "__main__":
    main()