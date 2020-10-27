import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from VIOLENCE_DETECTION.violenceDataset import ViolenceDataset
from VIOLENCE_DETECTION.datasetsMemoryLoader import crime2localLoadData, get_Test_Data, hockeyLoadData, hockeyTrainTestSplit, get_Fold_Data
from SALIENCY.saliencyModel import build_saliency_model
from LOCALIZATION.localization_utils import computeBoundingBoxFromMask, personDetectionInFrameYolo
from UTIL.util import min_max_normalize_tensor, min_max_normalize_np, load_torch_checkpoint
from VIDEO_REPRESENTATION.preprocessor import Preprocessor
from VIDEO_REPRESENTATION.imageAnalysis import show
from YOLOv3.yolo_inference import initializeYoloV3
from LOCALIZATION.bounding_box import BoundingBox
from LOCALIZATION.point import Point

from VIOLENCE_DETECTION.transforms import ucf2CrimeTransforms, hockeyTransforms
from constants import DEVICE
import constants

import torch
import torchvision
import torchvision.transforms as transforms
import cv2
from operator import itemgetter
from PIL import Image
import numpy as np
import re

from LOCALIZATION.localization_utils import findContours, bboxes_from_contours, process_mask, filterClosePersonsInFrame, joinBBoxes, IOU, joinBBoxes, intersetionArea
from LOCALIZATION.MaskRCNN import personDetectionInFrameMaskRCNN
import itertools
import copy



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
        # print('video={}, frame={}, frame_number={}, gt={}'.format(video_name, frame_name, frame_number, frame_data))
        if frame_number != int(frame_data[5]):
            print('=========*********** Error en Ground Truth!!!!!!!!!')
            break
        bb = BoundingBox(Point(int(frame_data[1]), int(frame_data[2])), Point(int(frame_data[3]), int(frame_data[4])))
        gt_bboxes.append(bb)
    
    one_box = None
    for gtb in gt_bboxes:
        if one_box is None:
            one_box = gtb 
        else:
            xmin = min(bbox1.pmin.x, bbox2.pmin.x)
            ymin = min(bbox1.pmin.y, bbox2.pmin.y)
            xmax = max(bbox1.pmax.x, bbox2.pmax.x)
            ymax = max(bbox1.pmax.y, bbox2.pmax.y)
    return gt_bboxes, one_box

def yolo():
    img_size = 416
    weights_path = "/Users/davidchoqueluqueroman/Desktop/PAPERS-CODIGOS/violencedetection2/YOLOv3/weights/yolov3.weights"
    class_path = "/Users/davidchoqueluqueroman/Desktop/PAPERS-CODIGOS/violencedetection2/YOLOv3/data/coco.names"
    model_def = "/Users/davidchoqueluqueroman/Desktop/PAPERS-CODIGOS/violencedetection2/YOLOv3/config/yolov3.cfg"
    person_model, classes = initializeYoloV3(img_size, class_path, model_def, weights_path, DEVICE)
    return person_model, classes

def refinement(persons_in_frame, saliency_bboxes, iou_threshold=0.3):
    # persons_filtered.sort(key=lambda x: x.iou, reverse=True)
    temporal_ious_regions = []
    anomalous_regions = []
    # personBox = persons_filtered[0]
    for personBox in persons_in_frame:
        for saliencyBox in saliency_bboxes:
            ia = intersetionArea(personBox, saliencyBox)
            if ia > 0:
                jb=joinBBoxes(personBox, saliencyBox)
                # anomalous_regions.append(personBox)
                anomalous_regions.append(jb)
            print('PersonArea={}, SaliencyArea={}, IA={}'.format(personBox.area,saliencyBox.area,ia))
    
    output = []
    if len(anomalous_regions) > 1:
        for a, b in itertools.combinations(anomalous_regions, 2):
            iou = IOU(a, b)
            if iou > 0.4:
                output.append(joinBBoxes(a, b))
            else:
                output.append(a)
                output.append(b)
        
    return anomalous_regions, output

def localization_error_it(predictions, gt_bboxes):
    counter = 1
    for i, gtb in enumerate(gt_bboxes):
        if predictions[i] is not None:
            # iou = IOU(gtb, predictions[i])
            it = intersetionArea(gtb, predictions[i])
            if it < 1 or it > gtb.area:
                counter += 1
        else:
            counter += 1
    el = counter / len(gt_bboxes)
    return counter, len(gt_bboxes), el

def localization_error_iou(predictions, gt_bboxes, threshold):
    counter = 0
    ious=[]
    for i, gtb in enumerate(gt_bboxes):
        if predictions[i] is not None:
            iou = IOU(gtb, predictions[i])
            ious.append(iou)
            if iou < threshold:
                counter += 1
        else:
            counter += 1
    el = counter / len(gt_bboxes)
    return ious, counter, len(gt_bboxes), el

    
def base_dataset(dataset, fold):
    if dataset == 'UCFCRIME2LOCAL':
        mytransfroms = ucf2CrimeTransforms(224)
        X, y, numFrames = crime2localLoadData(min_frames=40)
        train_idx, test_idx = get_Fold_Data(fold)
        train_x = list(itemgetter(*train_idx)(X))
        train_y = list(itemgetter(*train_idx)(y))
        train_numFrames = list(itemgetter(*train_idx)(numFrames))
        test_x = list(itemgetter(*test_idx)(X))
        test_y = list(itemgetter(*test_idx)(y))
        test_numFrames = list(itemgetter(*test_idx)(numFrames))
    elif dataset == 'HOCKEY':
        mytransfroms = hockeyTransforms(224)
        datasetAll, labelsAll, numFramesAll = hockeyLoadData()
        train_x, train_y, train_numFrames, test_x, test_y, test_numFrames = hockeyTrainTestSplit('train-test-' + str(fold), datasetAll, labelsAll, numFramesAll)
    return test_x, test_y, test_numFrames, mytransfroms

def localization():
    mask_model = build_saliency_model(num_classes=2)
    mask_model.to(DEVICE)
    model_path = 'MaskModel_bbone=alexnetv2_Dts=UCFCRIME2LOCAL_NDI-len=1-40_AreaLoss=8.0_SmoothLoss=0.5_PreservLoss=0.3_AreaLoss2=0.3_epochs=30'
    file = os.path.join(constants.PATH_RESULTS,
                        'MASKING/checkpoints',
                        model_path)

    checkpoint = load_torch_checkpoint(file)
    mask_model.load_state_dict(checkpoint['model_state_dict'])
    folds = [1,2,3,4,5]
    folds_lerrors = []
    save = False

    #Save images
    save_folder = os.path.join(constants.PATH_RESULTS, 'MASKING', 'images', model_path)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    for fold in folds:
        test_x, test_y, test_numFrames, mytransfroms = base_dataset('UCFCRIME2LOCAL', fold=fold)
        transf = ucf2CrimeTransforms(224)
        dataset = ViolenceDataset(dataset=test_x,
                                    labels=test_y,
                                    numFrames=test_numFrames,
                                    spatial_transform=mytransfroms['val'],
                                    numDynamicImagesPerVideo=1,
                                    videoSegmentLength=40,
                                    positionSegment='begin',
                                    overlaping=0,
                                    frame_skip=0,
                                    skipInitialFrames=20,
                                    ppType=None)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
        w = 320
        h = 240
        preprocessor = Preprocessor(None)
        yolo_detector, classes = yolo()
        maskRcnnDetector = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        maskRcnnDetector.eval()
        img_size = 416
        conf_thres = 0.8
        
        nms_thres = 0.4
        predictions = []
        y = []

        for i, data in enumerate(dataloader):
            inputs, labels, video_names, _, paths = data
            # print('video: ', video_names[0])
            batch_size, timesteps, C, H, W = inputs.size() # (1, 1, 3, 224, 224)
            dimages = inputs.view(batch_size * timesteps, C, H, W)
            dimages = dimages.cpu().numpy()
            dimages = [min_max_normalize_np(img).transpose(1, 2, 0)for img in dimages]
            
            # print('==========, Label: ', labels, type(labels))
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
                    # saliency_bboxes, preprocesing_outs, contours, hierarchy = computeBoundingBoxFromMask(mascara)
                    bynary = preprocessor.binarize(mascara, thresh=0.3, maxval=1)
                    bynary_morph = process_mask(bynary)
                    
                    img_contuors, contours, hierarchy = findContours(bynary_morph.astype(np.uint8), remove_fathers=True)
                    img_bboxes, saliency_bboxes = bboxes_from_contours(img_contuors, contours)
                    

                    if len(saliency_bboxes):
                        saliency_bboxes.sort(key=lambda x: x.area, reverse=True)
                        predictions.append(saliency_bboxes[0])
                    else:
                        predictions.append(None)
                    
                    dimages_gray = cv2.cvtColor(dimages[i], cv2.COLOR_BGR2GRAY);
                    dimages_gray_bynary = preprocessor.binarize(dimages_gray, thresh=0.5, maxval=1)
                    show(wait=50,
                        dimage=dimages[i],
                        dimages_gray=dimages_gray,
                        dimages_gray_bynary=dimages_gray_bynary,
                        mask=mascara,
                        binary=bynary,
                        bynary_morph=bynary_morph,
                        bb=img_bboxes)
                    frames = load_dynamic_image_frames(paths[i])
                    gt_bboxes, one_box = load_localization_ground_truth(paths[i])
                    y.append(one_box)
                    if len(frames) != len(gt_bboxes):
                        print('---------FRames y gt no coinciden...{}/{}'.format(len(frames), len(gt_bboxes)))
                    
                    # persons_in_frame = []
                    # for k, fr in enumerate(frames):
                    #     # if k==int(len(frames)/2):
                    #     if k==int(len(frames)/2):
                    #         persons_in_frame, person_detection_time = personDetectionInFrameYolo(yolo_detector, img_size, conf_thres, nms_thres, classes, fr, DEVICE)
                            # mask_rcnn_threshold = 0.4
                            # persons_in_frame, person_detection_time = personDetectionInFrameMaskRCNN(maskRcnnDetector, fr, mask_rcnn_threshold)
                    results_imgs = []
                    frames_raw = copy.deepcopy(frames)
                    for k, fr in enumerate(frames):
                        if len(saliency_bboxes) > 0:
                            s_bbox = saliency_bboxes[0]
                            cv2.rectangle(fr, (int(s_bbox.pmin.x), int(s_bbox.pmin.y)), (int(s_bbox.pmax.x), int(s_bbox.pmax.y)), constants.yellow, 2)
                        else:
                            print('Fail!!!!!')
                        # for s_bbox in saliency_bboxes:
                        #     cv2.rectangle(fr, (int(s_bbox.pmin.x), int(s_bbox.pmin.y)), (int(s_bbox.pmax.x), int(s_bbox.pmax.y)), constants.yellow, 2)
                        # for person in persons_in_frame:
                        #     cv2.rectangle(fr, (int(person.pmin.x), int(person.pmin.y)), (int(person.pmax.x), int(person.pmax.y)), constants.green, 2)
                        
                        # for freg in all_regions:
                        #     cv2.rectangle(fr, (int(freg.pmin.x), int(freg.pmin.y)), (int(freg.pmax.x), int(freg.pmax.y)), constants.blue, 2)

                        # for freg in final_regions:
                        #     cv2.rectangle(fr, (int(freg.pmin.x), int(freg.pmin.y)), (int(freg.pmax.x), int(freg.pmax.y)), constants.magenta, 3)
                        cv2.rectangle(fr, (one_box.pmin.x, int(one_box.pmin.y)), (int(one_box.pmax.x), int(one_box.pmax.y)), constants.blue, 2)
                        # cv2.rectangle(fr, (gt_bboxes[k].pmin.x, int(gt_bboxes[k].pmin.y)), (int(gt_bboxes[k].pmax.x), int(gt_bboxes[k].pmax.y)), constants.red, 2)
                        cv2.imshow('image', fr)
                        if cv2.waitKey(50) & 0xFF == ord('q'):
                            break
                        results_imgs.append(fr)
                    if save:
                        _, nname = os.path.split(video_names[0])
                        cv2.imwrite(os.path.join(save_folder, nname + '(in).png'), frames_raw[int(len(frames_raw) / 2)])
                        cv2.imwrite(os.path.join(save_folder, nname + '(out).png'), results_imgs[int(len(results_imgs) / 2)])
                        cv2.imwrite(os.path.join(save_folder, nname + '(sm-raw).png'), 255 * bynary)
                        cv2.imwrite(os.path.join(save_folder, nname + '(sm-final).png'), 255*bynary_morph)
                        cv2.imwrite(os.path.join(save_folder, nname + '(bm).png'), 255*dimages_gray_bynary)
                        cv2.imwrite(os.path.join(save_folder, nname + '(di).png'), 255*dimages[i])
        ious, c, t, lerror = localization_error_iou(predictions, y, threshold=0.5)
        folds_lerrors.append(lerror)
        print('Fold={}-Localization error({}/{})={}, \n--->ious={}'.format(fold, c, t, lerror, ious))
        # print('Ious=',ious)
    print('5 fold Errors=', folds_lerrors)


if __name__ == "__main__":
    localization()

