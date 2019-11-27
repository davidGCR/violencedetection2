import sys
sys.path.insert(1, '/media/david/datos/PAPERS-SOURCE_CODE/violencedetection')
import torchvision.transforms as T
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import torchvision
import torch
import numpy as np
import cv2
import random
from point import Point
from bounding_box import BoundingBox

# model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
# model.eval()

def random_colour_masks(image):
  """
  random_colour_masks
    parameters:
      - image - predicted masks
    method:
      - the masks of each predicted object is given random colour for visualization
  """
  colours = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
  r = np.zeros_like(image).astype(np.uint8)
  g = np.zeros_like(image).astype(np.uint8)
  b = np.zeros_like(image).astype(np.uint8)
  r[image == 1], g[image == 1], b[image == 1] = colours[random.randrange(0,10)]
  coloured_mask = np.stack([r, g, b], axis=2)
  return coloured_mask

def personDetectionInFrameMaskRCNN(model, ioImage, threshold):
    masks, pred_boxes, pred_class = get_prediction(model, ioImage, threshold)
    persons = []
    for idx, clase in enumerate(pred_class):
        if clase == 'person':
            bbox = BoundingBox(Point(pred_boxes[idx][0][0], pred_boxes[idx][0][1]),Point(pred_boxes[idx][1][0], pred_boxes[idx][1][1]))
            persons.append(bbox)
    return persons
    

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

def get_prediction(model, img, threshold):
    """
    get_prediction
        parameters:
        - img_path - path of the input image
        method:
        - Image is obtained from the image path
        - the image is converted to image tensor using PyTorch's Transforms
        - image is passed through the model to get the predictions
        - masks, classes and bounding boxes are obtained from the model and soft masks are made binary(0 or 1) on masks
            ie: eg. segment of cat is made 1 and rest of the image is made 0
        
    """
    # img = Image.open(img_path)
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    pred = model([img])
    
    pred_score = list(pred[0]['scores'].detach().numpy())
    # print('pred_score: ',pred_score, len(pred_score))
    pred_t = []#indices
    for x in pred_score:
        if x > threshold: 
            pred_t.append(pred_score.index(x))
    # print('pred_t: ',len(pred_t), pred_t)
    pred_t = pred_t[-1]
    # print('pred_t val: ',pred_t)
    masks = (pred[0]['masks']>0.5).squeeze().detach().cpu().numpy()
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
    masks = masks[:pred_t+1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    return masks, pred_boxes, pred_class


def instance_segmentation_api(img_path, threshold=0.5, rect_th=1, text_size=0.5, text_th=1):
  """
  instance_segmentation_api
    parameters:
      - img_path - path to input image
    method:
      - prediction is obtained by get_prediction
      - each mask is given random color
      - each mask is added to the image in the ration 1:0.8 with opencv
      - final output is displayed
  """
  masks, boxes, pred_cls = get_prediction(img_path, threshold)
  img = cv2.imread(img_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  for i in range(len(masks)):
    rgb_mask = random_colour_masks(masks[i])
    img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
    cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th)
    cv2.putText(img,pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
  plt.figure(figsize=(20,30))
  plt.imshow(img)
  plt.xticks([])
  plt.yticks([])
  plt.show()

# img_path = 'YOLOv3/data/samples/frame805.jpg'
# img = Image.open(img_path)
# masks, pred_boxes, pred_class = get_prediction(img_path, 0.5)
# print(type(masks), type(pred_boxes), type(pred_class), pred_class, pred_boxes)

# transform = T.Compose([T.ToTensor()])
# img = transform(img)
# model = personDetectionInFrameMaskRCNN()
# pred = model([img])
# masks = (pred[0]['masks']>0.5).squeeze().detach().cpu().numpy()
# print(masks.shape)
# plt.imshow(masks[0], cmap='gray')
# plt.show()

# instance_segmentation_api(img_path, 0.75,  rect_th=3, text_size=3, text_th=3)
# instance_segmentation_api(img_path, 0.5)