from __future__ import division

from .models import Darknet
# import models
from .utils.utils import load_classes, non_max_suppression
from .utils.datasets import transforms

import os
import sys
import time
import datetime
import argparse

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
import numpy as np


def initializeYoloV3(img_size, class_path, model_def, weights_path):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set up model
    model = Darknet(model_def, img_size=img_size).to(device)

    if weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(weights_path))

    model.eval()  # Set in evaluation mode

    classes = load_classes(class_path)  # Extracts class labels from file
    #classes:  ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 
    #           'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 
    #           'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 
    return model, classes

def rescale_boxes(boxes, current_dim, original_shape):
    """ Rescales bounding boxes to the original shape """
    orig_h, orig_w = original_shape

    # current_dim_y, current_dim_x = current_dim
    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
    # Image height and width after padding is removed
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x
    # Rescale bounding boxes to dimension of original image
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes

# def main():
#     img_size = 416
#     weights_path = "weights/yolov3.weights"
#     class_path = "data/coco.names"
#     model_def = "/media/david/datos/PAPERS-SOURCE_CODE/MyCode/YOLOv3/config/yolov3.cfg"
#     image_folder = "data/samples"
#     batch_size = 1
#     n_cpu = 4
#     conf_thres = 0.8
#     nms_thres = 0.4
    
#     model, classes = initializeYoloV3(img_size, class_path, model_def, weights_path)

#     image_path = 'data/samples/frame645.jpg'#test image
#     print("\nPerforming object detection:")
#     prev_time = time.time()
#     img = preProcessImage(image_path, img_size)
#     detections = inference(model, img, conf_thres, nms_thres)
#     current_time = time.time()
#     inference_time = datetime.timedelta(seconds=current_time - prev_time)
#     print("\t+ Inference Time: %s" % (inference_time))

#     if detections is not None:
#         detections = rescale_boxes(detections[0], img_size, img.shape[:2])
#         x1, y1, x2, y2, conf, cls_conf, cls_pred = detections[0]
#         unique_labels = detections[:, -1].cpu().unique()
       
#         # plotDetection(image_path, x1, y1, x2, y2, conf, cls_conf, cls_pred, unique_labels, classes)
#         # plt.show()
    
def plotDetection(image_path, x1, y1, x2, y2, conf, cls_conf, cls_pred, unique_labels, classes):
    # Bounding-box colors
    # Create plot
    img = np.array(Image.open(image_path))
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    n_cls_preds = len(unique_labels)
    bbox_colors = random.sample(colors, n_cls_preds)
    box_w = x2 - x1
    box_h = y2 - y1
    color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
    bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")     
    ax.add_patch(bbox)
    plt.text(
        x1,
        y1,
        s=classes[int(cls_pred)],
        color="white",
        verticalalignment="top",
        bbox={"color": color, "pad": 0},
    )
    
        
        # plt.axis("off")
        # plt.gca().xaxis.set_major_locator(NullLocator())
        # plt.gca().yaxis.set_major_locator(NullLocator())
        # # filename = path.split("/")[-1].split(".")[0]
        # plt.savefig("output/dog.png", bbox_inches="tight", pad_inches=0.0)
        # plt.close()

def inference(model, input_imgs, conf_thres, nms_thres): #input_imgs:  <class 'torch.Tensor'> torch.Size([1, 3, 416, 416])
    # print('input_imgs: ', type(input_imgs),input_imgs.size())
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    # Configure input
    input_imgs = Variable(input_imgs.type(Tensor))
    # print('input_imgs: ', input_imgs.size())
    # Get detections
    with torch.no_grad():
        detections = model(input_imgs) #  torch.Size([1, 10647, 85])
        detections = non_max_suppression(detections, conf_thres, nms_thres)#list len = 1
        # print('detction non_max_suppression: ', type(detections[0]),detections[0].size())# 0: <class 'torch.Tensor'> torch.Size([3, 7])
    # print('detectioins inference: ', detections)
    if detections[0] is not None:
        detections = torch.cat(detections)
    else:
        detections = None
    return detections #inference torch.Size([4, 7])

def preProcessImage(npImage, img_size):
    # npImage = np.transpose(1,2,0)
    # npImage = Image.fromarray(npImage.astype('uint8'), 'RGB')
    # npImage = transforms.ToPILImage(npImage)
    
    img = transforms.ToTensor()(npImage)
    # print('preProcessImage: ', type(img), img.size())
    # Pad to square resolution
    img, _ = pad_to_square2(img, 0)
    # Resize
    img = resize2(img, img_size)
    # print('preProcessImage: ', type(img))
    return img.unsqueeze(0)

def pad_to_square2(img, pad_value):
    size = img.size()
    c = size[0]
    h = size[1]
    w = size[2]
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)
    return img, pad

def resize2(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

        
if __name__ == '__main__':
    main()