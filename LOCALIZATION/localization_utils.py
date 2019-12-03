import sys
sys.path.insert(1, '/media/david/datos/PAPERS-SOURCE_CODE/violencedetection')
import os
import constants
import glob
import numpy as np
import cv2
import math
from point import Point
from bounding_box import BoundingBox
from YOLOv3 import yolo_inference
import torch
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
from mpl_toolkits.axes_grid1 import ImageGrid
import copy
import itertools

# import point.Point as Point
def IOU(gt_bbox1, bbox2):
    if bbox2 == None:
        return 0
    # calculate area of intersection rectangle
    inter_area = intersetionArea(gt_bbox1,bbox2)
    # calculate area of actual and predicted boxes
    actual_area = gt_bbox1.area()
    pred_area = bbox2.area()
 
    # computing intersection over union
    iou = inter_area / float(actual_area + pred_area - inter_area)
 
    # return the intersection over union value
    return iou

def mAP(dataframe):
    # calculating Precision and recall
    Precision = []
    Recall = []

    TP = FP = 0
    FN = len(dataframe['tp/fp'] == 'TP')
    print('all TP', FN)
    for index , row in dataframe.iterrows():     
        if row.iou > 0.5:
            TP =TP+1
        else:
            FP =FP+1    
        try:
            AP = TP/(TP+FP)
            Rec = TP/(TP+FN)
        except ZeroDivisionError:
            AP = Recall = 0.0
        Precision.append(AP)
        Recall.append(Rec)
    dataframe['Precision'] = Precision
    dataframe['Recall'] = Recall
    dataframe['ip'] = dataframe.groupby('Recall')['Precision'].transform('max')
    prec_at_rec = []
    for recall_level in np.linspace(0.0, 1.0, 11):
        try:
            x = dataframe[dataframe['Recall'] >= recall_level]['Precision']
            prec = max(x)
        except:
            prec = 0.0
        prec_at_rec.append(prec)
    avg_prec = np.mean(prec_at_rec)
    print('11 point precision is ', prec_at_rec)
    print('\nmap is ', avg_prec)

def filterClosePersonsInFrame(personsBBoxes, thresh_close_persons):
    """Join persons bboxes if they iou is greter than a threshold"""
    persons_filtered = []
    only_joined_regions = []
    # if len(personsBBoxes) > 0:
    for p1, p2 in itertools.combinations(personsBBoxes, 2):
        # iou = IOU(p1, p2)
        iou = intersetionArea(p1, p2)
        iou = p1.percentajeArea(iou)
        # print('iou: ', iou, p1, p2)
        if iou >= thresh_close_persons:
            presult = joinBBoxes(p1,p2)
            # persons_filtered.append(p1)
            # persons_filtered.append(p2)
            persons_filtered.append(presult)
            only_joined_regions.append(presult)
        else:
            persons_filtered.append(p1)
            persons_filtered.append(p2)
            
    return persons_filtered, only_joined_regions

def plotOnlyBBoxOnImage(image, boxes, color, text, font_size):
    if boxes is not None:

        if isinstance(image, np.ndarray):
            image = (image * 255 / np.max(image)).astype('uint8')
            image = Image.fromarray(image)
            # print('image pil: ', image.size)
            # draw = ImageDraw.Draw()
        
        draw = ImageDraw.Draw(image)
        # h = box.pmax.y - box.pmin.y
        # w = box.pmax.x - box.pmin.x
        font = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', font_size)
        if isinstance(boxes, list):
            for box in boxes:
                # print('******* box: ', box)
                draw.rectangle((box.pmin.x, box.pmin.y, box.pmax.x, box.pmax.y), fill=None, outline=color)
                draw.text((box.pmin.x, box.pmin.y), text, fill =color,font = font, align ="left")
        else:
            draw.rectangle((boxes.pmin.x, boxes.pmin.y, boxes.pmax.x, boxes.pmax.y), fill=None, outline=color)
            draw.text((boxes.pmin.x, boxes.pmin.y), text, fill=color,font = font, align ="left")  
            
        # drawing text size 
        
        # rect = patches.Rectangle((box.pmin.x, box.pmin.y), w, h, linewidth=1, edgecolor=color, facecolor='none')
        # # Add the patch to the Axes
        # ax.add_patch(rect)
        # plt.text(box.pmin.x, box.pmin.y, s=text, color="white", verticalalignment="top", bbox={"color": color, "pad": 0},)
        # fig.patches.extend([plt.Rectangle((box.pmin.x, box.pmin.y),w,h, color=color, alpha=0.5,zorder=1000,figure=fig)])
    return image

def findAnomalyRegionsOnFrame(personBboxes, saliencyBboxes, iou_threshold):
    """Detect anomalous regions in one frame: Iou between persons and saliency"""
    anomalyRegions = []
    # person - saliency regions intersection
    for personBox in personBboxes:
        for saliencyBox in saliencyBboxes:
            iou = IOU(personBox, saliencyBox)
            # iou = intersetionArea(personBox, saliencyBox)
            # intersection = personBox.percentajeArea(iou)
            print('***iou: ', iou)
            if iou >= iou_threshold:
                personBox.iou = iou
                anomalyRegions.append(personBox)
    
    
    # while len(anomalyRegions) > 1:
    #     filtered_regions = []
    #     for p1, p2 in itertools.combinations(anomalyRegions, 2):
    #         if verifyClosePersons(p1, p2, max_dist_persons_threshold):
    #             joinBBox = joinBBoxes(p1, p2)
    #             filtered_regions.append(joinBBox)
    #     anomalyRegions = list(set(filtered_regions.copy()))
    # if len(anomalyRegions) < 1:
    #     return None
    return anomalyRegions

def verifyClosePersons(p1, p2, d_treshold):
    dist = distance(p1.center, p2.center)
    if dist < d_treshold:
        return True
    return False


def intersetionArea(bbox1, bbox2): 
    dx = min(bbox1.pmax.x, bbox2.pmax.x) - max(bbox1.pmin.x, bbox2.pmin.x)
    dy = min(bbox1.pmax.y, bbox2.pmax.y) - max(bbox1.pmin.y, bbox2.pmin.y)
    if (dx>=0) and (dy>=0):
        return dx * dy
    else: return 0



def joinBBoxes(bbox1, bbox2):
    xmin = min(bbox1.pmin.x, bbox2.pmin.x)
    ymin = min(bbox1.pmin.y, bbox2.pmin.y)
    xmax = max(bbox1.pmax.x, bbox2.pmax.x)
    ymax = max(bbox1.pmax.y, bbox2.pmax.y)
    bbox = BoundingBox(Point(xmin, ymin), Point(xmax, ymax))
    return bbox

def getFramesFromSegment(video_name, frames_segment, num_frames):
    # print('sdfasffffffffffffffffffffffffff')
    names = []
    frames = []
    bboxes = []
    if num_frames == 'all':
        for frame_info in frames_segment:
            # f_info = frame_info[]
            frame_name = str(frame_info[0][0])
            # print(frame_info, len(frame_info))
            frame_path = os.path.join(video_name, frame_name)
            names.append(frame_path)
            # image = np.array(Image.open(frame_path))
            image = Image.open(frame_path)
            frames.append(image)
            bbox = BoundingBox(Point(frame_info[constants.IDX_XMIN].float(), frame_info[constants.IDX_YMIN].float()), Point(frame_info[constants.IDX_XMAX].float(), frame_info[constants.IDX_YMAX].float()))
            bboxes.append(bbox)
    elif num_frames == 'first':
        frame_info = frames_segment[0]
        frame_name = str(frame_info[0][0])
        frame_path = os.path.join(video_name, frame_name)
        names.append(frame_path)
        image = Image.open(frame_path)
        frames.append(image)
        bbox = BoundingBox(Point(frame_info[constants.IDX_XMIN], frame_info[constants.IDX_YMIN]), Point(frame_info[constants.IDX_XMAX], frame_info[constants.IDX_YMAX]))
        bboxes.append(bbox)
    elif  num_frames == 'extremes':
        frame_info_first = frames_segment[0]
        frame_name_first = str(frame_info_first[0][0])
        frame_path_first = os.path.join(video_name, frame_name_first)
        names.append(frame_path_first)
        image = Image.open(frame_path_first)
        frames.append(image)
        bbox = BoundingBox(Point(frame_info_first[constants.IDX_XMIN], frame_info_first[constants.IDX_YMIN]), Point(frame_info_first[constants.IDX_XMAX], frame_info_first[constants.IDX_YMAX]))
        bboxes.append(bbox)

        frame_info_end = frames_segment[len(frames_segment)-1]
        frame_name_end = str(frame_info_end[0][0])
        frame_path_end = os.path.join(video_name, frame_name_end)
        names.append(frame_path_end)
        image = Image.open(frame_path_end)
        frames.append(image)
        bbox = BoundingBox(Point(frame_info_end[constants.IDX_XMIN], frame_info_end[constants.IDX_YMIN]), Point(frame_info_end[constants.IDX_XMAX], frame_info_end[constants.IDX_YMAX]))
        bboxes.append(bbox)
    return names, frames, bboxes


def personDetectionInSegment(frames_list, yolo_model, img_size, conf_thres, nms_thres, classes, type_num_frames):
    bbox_persons_in_segment = []
    if type_num_frames == 'all':
        for ioImage in frames_list:
            bbox_persons_in_frame = personDetectionInFrame(yolo_model, img_size, conf_thres, nms_thres, classes, ioImage)
            bbox_persons_in_segment.append(bbox_persons_in_frame)
    elif type_num_frames == 'first':
        bbox_persons_in_frame = personDetectionInFrame(yolo_model, img_size, conf_thres, nms_thres, classes, frames_list[0])
        bbox_persons_in_segment.append(bbox_persons_in_frame)
    elif  type_num_frames == 'extremes':
        bbox_persons_in_frame_first = personDetectionInFrame(yolo_model, img_size, conf_thres, nms_thres, classes, frames_list[0])
        bbox_persons_in_frame_end = personDetectionInFrame(yolo_model, img_size, conf_thres, nms_thres, classes, frames_list[len(frames_list)-1])
        bbox_persons_in_segment.append(bbox_persons_in_frame_first)
        bbox_persons_in_segment.append(bbox_persons_in_frame_end)
    return bbox_persons_in_segment


def personDetectionInFrame(model, img_size, conf_thres, nms_thres, classes, ioImage, plot = False):
    # print('='*20+' YOLOv3 - ', frame_path)
    img = yolo_inference.preProcessImage(ioImage, img_size)
    detections = yolo_inference.inference(model, img, conf_thres, nms_thres)
    ioImage = np.array(ioImage)
    # image = np.array(Image.open(frame_path))
    # print('image type: ', type(image), image.dtype, image.shape)
    if plot:
        fig, ax = plt.subplots()
        ax.imshow(ioImage)
    bbox_persons = []
    if detections is not None:
        # print('detectios rescale: ', type(detections), detections.size())
        detections = yolo_inference.rescale_boxes(detections, 416, ioImage.shape[:2])
        unique_labels = detections[:, -1].cpu().unique()
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            if classes[int(cls_pred)] == 'person':
                pmin = Point(x1, y1)
                pmax = Point(x2,y2)
                bbox_persons.append(BoundingBox(pmin,pmax))
            # print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
            if plot:
                box_w = x2 - x1
                box_h = y2 - y1
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor='g', facecolor="none")
                ax.add_patch(bbox)
                plt.text( x1, y1, s=classes[int(cls_pred)], color="white", verticalalignment="top", bbox={"color": 'r', "pad": 0}, )
    return bbox_persons

def distance(point1, point2):
    distance = math.sqrt(((point1.x - point2.x)** 2) + ((point1.y - point2.y)** 2))
    return distance

def computeBoundingBoxFromMask(mask):
    """
    *** mask: rgb numpy image
    """
    mask = thresholding_cv2(mask)
    img_process_mask = process_mask(mask)
    img_contuors, contours = findContours(img_process_mask, remove_fathers=True)
    img_bboxes, bboxes = bboxes_from_contours(img_contuors, contours)
    preprocesing_reults = {'mask':mask, 'process_mask':img_process_mask, 'contours':img_contuors, 'boxes': img_bboxes}
    return bboxes, preprocesing_reults

def process_mask(img):
    kernel_exp = np.ones((5, 5), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_exp)
    kernel_dil = np.ones((7, 7), np.uint8)
    img = cv2.dilate(img, kernel_dil, iterations=1)
    kernel_clo = np.ones((11, 11), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_clo)
    return img

def findContours(img, remove_fathers = True):
    # Detect edges using Canny
    # canny_output = cv2.Canny(src_gray, threshold, threshold * 2)
    # Find contours
    # color = cv2.Scalar(0, 255, 0)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Draw contours
    img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    if remove_fathers:
        removed = []
        for idx,contour in enumerate(contours):
            if hierarchy[0, idx, 3] == -1:
                removed.append(contour)
        contours = removed
    for i in range(len(contours)):
        cv2.drawContours(img, contours, i, (0, 255, 0), 2, cv2.LINE_8, hierarchy, 0)
    return img, contours



def bboxes_from_contours(img, contours):
    contours_poly = [None]*len(contours)
    boundRect = [None] * len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
    image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    bboxes = []
    for i, rect in enumerate(boundRect):
        # print('REct: ', rect)
        bb = cvRect2BoundingBox(rect)
        bboxes.append(bb)
        color_red = (0,0,255)
        # cv2.drawContours(drawing, contours_poly, i, color)
        cv2.rectangle(image, (int(boundRect[i][0]), int(boundRect[i][1])), (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color_red, 2)
    return image, bboxes

def cvRect2BoundingBox(cvRect):
    pmin = Point(cvRect[0], cvRect[1])
    pmax = Point(cvRect[0]+cvRect[2], cvRect[1]+cvRect[3])
    bb = BoundingBox(pmin,pmax)
    # print('bbbbbbbbxxxxx: ', pmin.x, bb.center.x)
    return bb

def plot_grid(fig, images):
    """
    A grid of 2x2 images with 0.05 inch pad between images and only
    the lower-left axes is labeled.
    """
    grid = ImageGrid(fig, (1,1,1),  # similar to subplot(141)
                     nrows_ncols=(1, len(images)),
                     axes_pad=0.05,
                     label_mode="1",
                     )

    # Z, extent = get_demo_image()
    for ax, image  in zip(grid, images):
        ax.imshow(image)

    # This only affects axes in first column and second row as share_all =
    # False.
    # grid.axes_llc.set_xticks([-2, 0, 2])
    # grid.axes_llc.set_yticks([-2, 0, 2])

def thresholding_cv2(x):
        x = 255*x #between 0-255
        x = x.astype('uint8')
        # th = cv2.adaptiveThreshold(x,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,2)
        # Otsu's thresholding
        x = cv2.GaussianBlur(x,(5,5),0)
        # print('x numpy: ', x.shape, x.dtype)
        ret2, th = cv2.threshold(x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return th

def normalize_tensor(self, img):
        # print("normalize:", img.size())
        _min = torch.min(img)
        _max = torch.max(img)
        # print("min:", _min.item(), ", max:", _max.item())
        return (img - _min) / (_max - _min)

def get_anomalous_video(video_test_name, reduced_dataset = True):
    """ get anomalous video """
    label = video_test_name[:-3]
    path = os.path.join(constants.PATH_UCFCRIME2LOCAL_FRAMES_REDUCED, video_test_name) if reduced_dataset else os.path.join(constants.PATH_UCFCRIME2LOCAL_FRAMES, video_test_name)
    
    list_frames = os.listdir(path) 
    list_frames.sort()
    num_frames = len(glob.glob1(path, "*.jpg"))
    bdx_file_path = os.path.join(constants.PATH_UCFCRIME2LOCAL_BBOX_ANNOTATIONS, video_test_name+'.txt')
    data = []
    with open(bdx_file_path, 'r') as file:
        for row in file:
            data.append(row.split())
    data = np.array(data)
    bbox_infos_frames = []
    for frame in list_frames:
        num_frame = int(frame[len(frame) - 7:-4])
        if num_frame != int(data[num_frame, 5]):
            sys.exit('Houston we have a problem: index frame does not equal to the bbox file!!!')
            # print('Houston we have a problem: index frame does not equal to the bbox file!!!')
        flac = int(data[num_frame,6]) # 1 if is occluded: no plot the bbox
        xmin = int(data[num_frame, 1])
        ymin= int(data[num_frame, 2])
        xmax = int(data[num_frame, 3])
        ymax = int(data[num_frame, 4])
        info_frame = [frame, flac, xmin, ymin, xmax, ymax]
        bbox_infos_frames.append(info_frame)
    
    return path, label, bbox_infos_frames, num_frames

def tensor2numpy(x):
    x = x / 2 + 0.5
    x = x.numpy()
    x = np.transpose(x, (1, 2, 0))
    # print('x: ', type(x), x.shape)
    return x

def rgb2grayUnrepeat(x):
    x = x[:,:, 0]
    return x

def gray2rgbRepeat(x):
    x = np.stack([x, x, x], axis=2)
    return x