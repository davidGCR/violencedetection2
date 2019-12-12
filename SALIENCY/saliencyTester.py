import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from .saliencyModel  import build_saliency_model
import random
from operator import itemgetter
from loss import Loss
import os
from torchvision.utils import save_image
import constants
import torchvision.transforms as transforms
import cv2
import LOCALIZATION.tracker as tracker

class SaliencyTester():
    def __init__(self,saliency_model_file, num_classes, dataloader, datasetAll, input_size, saliency_config, numDiPerVideos, threshold):
        self.saliency_model_file = saliency_model_file
        self.num_classes = num_classes
        self.dataloader = dataloader
        self.datasetAll = datasetAll
        self.input_size = input_size
        self.saliency_config = saliency_config
        self.numDiPerVideos = numDiPerVideos
        self.threshold = threshold
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = build_saliency_model(num_classes=num_classes)
        self.net = self.net.to(device)
        self.net = torch.load(saliency_model_file)
        self.net.eval()

    def compute_sal_map(self, data):
        # """Compute mask from dinamyc image"""
          # dataset load [bs,ndi,c,w,h]
        # print('video: ',video_name[0])
        # tracker.plotBoundingBoxSegment(video_name[0],bbox_segments)
        # print( "inputs, labels: ", type(di_images), di_images.size(), type(labels), labels.size() )
        di_images, labels, video_name, bbox_segments = data
        if self.numDiPerVideos > 1:
            di_images = torch.squeeze(di_images, 0)  # get one di [ndi,c,w,h
        # if numDiPerVideos>1:
        #     inputs = inputs.permute(1, 0, 2, 3, 4)
        #     inputs = torch.squeeze(inputs, 0)  #get one di [bs,c,w,h]
        di_images, labels = Variable(di_images.cuda()), Variable(labels.cuda())
        masks, _ = self.net(di_images, labels)
        return di_images, masks

    def compute_mask(self, di_images, labels):
        """Compute mask from dinamyc image"""
          # dataset load [bs,ndi,c,w,h]
        if self.numDiPerVideos > 1:
            di_images = torch.squeeze(di_images, 0)  # get one di [ndi,c,w,h
        # if numDiPerVideos>1:
        #     inputs = inputs.permute(1, 0, 2, 3, 4)
        #     inputs = torch.squeeze(inputs, 0)  #get one di [bs,c,w,h]
        di_images, labels = Variable(di_images.cuda()), Variable(labels.cuda())
        masks, _ = self.net(di_images, labels)
        return masks


    def thresholding_cv2(self, x, rgb=True):
        x = 255*x #between 0-255
        x = x.astype('uint8')
        # th = cv2.adaptiveThreshold(x,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,2)
        # Otsu's thresholding
        x = cv2.GaussianBlur(x,(5,5),0)
        # print('x numpy: ', x.shape, x.dtype)
        ret2, th = cv2.threshold(x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if rgb:
            th = np.stack([th, th, th], axis=2)
        # plt.imshow(th)
        
        return th

    def repeat_channels3(self, img):
        img = np.stack([img, img, img], axis=2)
        return img

    def histogram(self,x):
        img1 = x / 2 + 0.5
        npimg1 = img1.numpy()
        plt.hist(npimg1.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
        plt.show()
    
    def plot_sample(self, images, title, save=False):
        # print('tensor: ', type(img1), img1.size())
        num_images = len(images) 

        fig2 = plt.figure(figsize=(12, 12))
        fig2.suptitle(title, fontsize=16)
        for idx, img in enumerate(images):
            print('figure size: ',img.size())
            imag = img / 2 + 0.5
            imag = imag.numpy()
            ax = plt.subplot(num_images, 1, idx + 1)
            plt.imshow(np.transpose(imag, (1, 2, 0)))
        plt.show()
        print(title)
        if save:
            fig2.savefig(
                os.path.join(
                    "/media/david/datos/Violence DATA/HockeyFights/saliencyResults",
                    title + ".png",
                )
            )

    def min_max_normalize_tensor(self, img):
        # print("normalize:", img.size())
        _min = torch.min(img)
        _max = torch.max(img)
        # print("min:", _min.item(), ", max:", _max.item())
        return (img - _min) / (_max - _min)
    
    def normalize_ndarray(self, img):
        _min = np.amin(img)
        _max = np.amax(img)
        # print("min:", _min, ", max:", _max)
        return (img - _min) / (_max - _min)

    def imshow(self, img):
        img = img / 2 + 0.5
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    # def thresholding_torch(self, x, threshold):
    #     self.histogram(x)
    #     x = self.normalize(x)
    #     # self.histogram(x)
    #     # histogram(x)
    #     # t = transforms.Compose([
    #     #     transforms.ToPILImage(),
    #     #     transforms.Grayscale(num_output_channels=3),
    #     #     transforms.ToTensor()])
    #     x = torch.squeeze(x, 0)
    #     # x = x.permute(1, 2, 0)
    #     # x = t(x)
    #     x = x > threshold
    #     x = x.float()
    #     x = x.repeat(3, 1, 1)
    #     print('X size out : ', x.size())
    #     # print("xmax:")
    #     return x

    
# def test( saliency_model_file, num_classes, dataloader, datasetAll, input_size, saliency_config, numDiPerVideos, threshold ):
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     net = saliencyModel.build_saliency_model(num_classes=num_classes)
#     net = net.to(device)
#     net = torch.load(saliency_model_file)
#     net.eval()
#     padding = 10
#     # cuda0 = torch.device('cuda:0')
#     ones = torch.ones(1, input_size, input_size)
#     zeros = torch.zeros(1, input_size, input_size)
#     ones = ones.to(device)
#     zeros = zeros.to(device)
#     images = []
#     # plt.figure(1)
#     for i, data in enumerate(dataloader, 0 ):  # inputs, labels:  <class 'torch.Tensor'> torch.Size([3, 3, 224, 224]) <class 'torch.Tensor'> torch.Size([3])
#         print("-" * 150)
#         images = []
#         inputs, labels, video_name, bbox_segments = data  # dataset load [bs,ndi,c,w,h]
#         print('video: ',video_name[0])
#         tracker.plotBoundingBoxSegment(video_name[0],bbox_segments)
#         print(
#             "inputs, labels: ", type(inputs), inputs.size(), type(labels), labels.size()
#         )
#         if numDiPerVideos > 1:
#             inputs = torch.squeeze(inputs, 0)  # get one di [ndi,c,w,h
#         # if numDiPerVideos>1:
#         #     inputs = inputs.permute(1, 0, 2, 3, 4)
#         #     inputs = torch.squeeze(inputs, 0)  #get one di [bs,c,w,h]
#         inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
#         masks, _ = net(inputs, labels)
#         # print('masks raw: ',masks.size())
#         # y = torch.where(masks > masks.view(masks.size(0), masks.size(1), -1).mean(2)[:, :, None, None], ones, zeros)
#         di_images = torchvision.utils.make_grid(inputs.cpu().data, padding=padding)
#         mask_images = torchvision.utils.make_grid(masks.cpu().data, padding=padding)
#         segmented_images = torchvision.utils.make_grid((inputs * masks).cpu().data, padding=padding )  # apply soft mask
#         # binary_masks = thresholding(masks.cpu().data,threshold)
#         # binary_masks = thresholding_cv2(masks.cpu().data,threshold)
#         images.append(di_images)
#         images.append(mask_images)
#         images.append(segmented_images)
#         # images.append(binary_masks)
#         plot_sample(images,video_name)



