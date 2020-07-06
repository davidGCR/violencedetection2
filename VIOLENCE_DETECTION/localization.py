import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from VIOLENCE_DETECTION.violenceDataset import ViolenceDataset
from VIOLENCE_DETECTION.datasetsMemoryLoader import crime2localLoadData, get_Test_Data
from SALIENCY.saliencyModel import build_saliency_model
from LOCALIZATION.localization_utils import computeBoundingBoxFromMask, personDetectionInFrameYolo
from UTIL.util import min_max_normalize_tensor, min_max_normalize_np
from VIDEO_REPRESENTATION.preprocessor import Preprocessor
from VIDEO_REPRESENTATION.imageAnalysis import show
from YOLOv3.yolo_inference import initializeYoloV3
from LOCALIZATION.bounding_box import BoundingBox
from LOCALIZATION.point import Point
from VIOLENCE_DETECTION.transforms import ucf2CrimeTransforms
from constants import DEVICE
import constants

import torch
import torchvision.transforms as transforms
import cv2
from operator import itemgetter
from PIL import Image
import numpy as np
import re

def load_dynamic_image_frames(paths):
    images = []
    for frame_path in paths:
        # print(frame_path, type(frame_path))
        frame = Image.open(frame_path[0])
        frame = np.array(frame)
        images.append(frame)
    return images

def load_localization_ground_truth(paths):
    pth, _ = os.path.split(paths[0][0])
    _, video_name = os.path.split(pth)
    video_name = video_name[:-8]
    bdx_file_path = os.path.join(constants.PATH_UCFCRIME2LOCAL_Txt_ANNOTATIONS, video_name+'.txt')
    data = [] 
    with open(bdx_file_path, 'r') as file:
        for row in file:
            data.append(row.split())
    # data = np.array(data)
    gt_bboxes = []
    for i,frame_path in enumerate(paths):
        pth, frame_name = os.path.split(frame_path[0])
        # _, video_name = os.path.split(pth)
        splits = re.split('(\d+)', frame_name)
        frame_number = int(splits[1])
        frame_data = data[frame_number]
        print('video={}, frame={}, frame_number={}, gt={}'.format(video_name, frame_name, frame_number, frame_data))
        if frame_number != int(frame_data[5]):
            print('=========*********** Error en Ground Truth!!!!!!!!!')
            break
        bb = BoundingBox(Point(int(frame_data[1]), int(frame_data[2])), Point(int(frame_data[3]), int(frame_data[4])))
        gt_bboxes.append(bb)
    return gt_bboxes

def yolo():
    img_size = 416
    weights_path = "/Users/davidchoqueluqueroman/Desktop/PAPERS-CODIGOS/violencedetection2/YOLOv3/weights/yolov3.weights"
    class_path = "/Users/davidchoqueluqueroman/Desktop/PAPERS-CODIGOS/violencedetection2/YOLOv3/data/coco.names"
    model_def = "/Users/davidchoqueluqueroman/Desktop/PAPERS-CODIGOS/violencedetection2/YOLOv3/config/yolov3.cfg"
    person_model, classes = initializeYoloV3(img_size, class_path, model_def, weights_path, DEVICE)
    return person_model, classes

def localization():
    mask_model = build_saliency_model(num_classes=2)
    mask_model.to(DEVICE)
    file = os.path.join(constants.PATH_RESULTS,
                        'MASKING/checkpoints',
                        'MaskModel_backnone=resnet50_NDI-len=1-40_AreaLoss=8_SmoothLoss=0.5_PreservLoss=0.3_AreaLoss2=0.3_epochs=25_epoch=23_loss=2.1115.pth')

    if DEVICE == 'cuda:0':
        mask_model.load_state_dict(torch.load(file), strict=False)
    else:
        mask_model.load_state_dict(torch.load(file, map_location=DEVICE))
    X, y, numFrames = crime2localLoadData(min_frames=0)
    test_idx = get_Test_Data(fold=1)
    test_x = list(itemgetter(*test_idx)(X))
    test_y = list(itemgetter(*test_idx)(y))
    test_numFrames = list(itemgetter(*test_idx)(numFrames))
    # transf = transforms.Compose(
    #         [
    #             transforms.CenterCrop(224),
    #             transforms.ToTensor()   
    #         ])
    transf = ucf2CrimeTransforms(224)
    dataset = ViolenceDataset(dataset=test_x,
                                labels=test_y,
                                numFrames=test_numFrames,
                                spatial_transform=transf['val'],
                                numDynamicImagesPerVideo=1,
                                videoSegmentLength=40,
                                positionSegment='begin',
                                overlaping=0,
                                frame_skip=0,
                                skipInitialFrames=10,
                                ppType=None)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    w = 320
    h = 240
    preprocessor = Preprocessor(None)
    yolo_detector, classes = yolo()
    img_size = 416
    conf_thres = 0.8
    
    nms_thres = 0.4
    for i, data in enumerate(dataloader):
        inputs, labels, video_names, _, paths = data
        print('Inputs: ', inputs.size())
        batch_size, timesteps, C, H, W = inputs.size() # (1, 1, 3, 224, 224)
        dimages = inputs.view(batch_size * timesteps, C, H, W)
        dimages = dimages.cpu().numpy()
        dimages = [min_max_normalize_np(img).transpose(1, 2, 0)for img in dimages]
        
        print('==========, Label: ', labels, type(labels))
        # print('paths=', i, len(paths), len(paths[0]), paths)
        if labels.item() == 1:
            mascaras = []
            with torch.no_grad():
                masks, _ = mask_model(inputs, labels)
                numMask, c, hh, ww = masks.size()
            for i in range(numMask):
                mascara = masks[i].detach().cpu()  #(1, 1, 224, 224)
                # print('mascara=', mascara.size())
                # mascara = torch.squeeze(mascara, 0)  #(1, 224, 224)
                mascara = min_max_normalize_tensor(mascara)  #to 0-1
                mascara = mascara.numpy().transpose(1, 2, 0)
                mascara = cv2.resize(mascara, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
                saliency_bboxes, preprocesing_outs, contours, hierarchy = computeBoundingBoxFromMask(mascara)
                bynary = preprocessor.binarize(mascara, thresh=0.5, maxval=1)
                
                
                
                show(wait=50, dimage=dimages[i], mask=mascara, binary=bynary)
                frames = load_dynamic_image_frames(paths[i])
                gt_bboxes = load_localization_ground_truth(paths[i])
                if len(frames) != len(gt_bboxes):
                    print('---------FRames y gt no coinciden...{}/{}'.format(len(frames), len(gt_bboxes)))
                for k, fr in enumerate(frames):
                #     persons_in_frame, person_detection_time = personDetectionInFrameYolo(yolo_detector, img_size, conf_thres,nms_thres, classes, fr, DEVICE)
                #     for s_bbox in saliency_bboxes:
                #         cv2.rectangle(fr, (int(s_bbox.pmin.x), int(s_bbox.pmin.y)), (int(s_bbox.pmax.x), int(s_bbox.pmax.y)), constants.yellow, 2)
                #     for person in persons_in_frame:
                #         cv2.rectangle(fr, (int(person.pmin.x), int(person.pmin.y)), (int(person.pmax.x), int(person.pmax.y)), constants.green, 2)
                    cv2.rectangle(fr, (gt_bboxes[k].pmin.x, int(gt_bboxes[k].pmin.y)), (int(gt_bboxes[k].pmax.x), int(gt_bboxes[k].pmax.y)), constants.red, 2)
                    cv2.imshow('image', fr)
                    # cv2.imshow('dimage', dimages[0])
                    if cv2.waitKey(50) & 0xFF == ord('q'):
                        break

if __name__ == "__main__":
    localization()

