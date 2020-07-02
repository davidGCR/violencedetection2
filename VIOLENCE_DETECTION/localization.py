import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from VIOLENCE_DETECTION.violenceDataset import ViolenceDataset
from VIOLENCE_DETECTION.datasetsMemoryLoader import crime2localLoadData, get_Test_Data
from SALIENCY.saliencyModel import build_saliency_model
from LOCALIZATION.localization_utils import computeBoundingBoxFromMask
from UTIL.util import min_max_normalize_tensor
from VIDEO_REPRESENTATION.preprocessor import Preprocessor
from VIDEO_REPRESENTATION.imageAnalysis import show
from constants import DEVICE
import constants

import torch
import cv2
from operator import itemgetter
from PIL import Image
import numpy as np

def load_dynamic_image_frames(paths):
    images = []
    for frame_path in paths:
        # print(frame_path, type(frame_path))
        frame = Image.open(frame_path[0])
        frame = np.array(frame)
        images.append(frame)
    return images

def localization():
    mask_model = build_saliency_model(num_classes=2)
    mask_model.to(DEVICE)
    file = os.path.join(constants.PATH_RESULTS,
                        'MASKING/checkpoints',
                        'MaskModel_backnone=resnet50_NDI=1_AreaLoss=8_SmoothLoss=0.5_PreservLoss=0.3_AreaLoss2=0.3_epochs=10-epoch-9.pth')

    if DEVICE == 'cuda:0':
        mask_model.load_state_dict(torch.load(file), strict=False)
    else:
        mask_model.load_state_dict(torch.load(file, map_location=DEVICE))
    X, y, numFrames = crime2localLoadData(min_frames=0)
    test_idx = get_Test_Data(fold=4)
    test_x = list(itemgetter(*test_idx)(X))
    test_y = list(itemgetter(*test_idx)(y))
    test_numFrames = list(itemgetter(*test_idx)(numFrames))
    dataset = ViolenceDataset(dataset=test_x,
                                labels=test_y,
                                numFrames=test_numFrames,
                                spatial_transform=None,
                                numDynamicImagesPerVideo=4,
                                videoSegmentLength=10,
                                positionSegment='begin',
                                overlaping=0,
                                frame_skip=2,
                                skipInitialFrames=10,
                                ppType=None)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    w = 320
    h = 240
    preprocessor = Preprocessor(None)
    for i, data in enumerate(dataloader):
        inputs, labels, video_names, _, paths = data
        
        # print('inputs=', i, inputs.size())
        print('==========, Label: ', labels, type(labels))
        # print('paths=', i, len(paths), len(paths[0]), paths)
        mascaras = []
        with torch.no_grad():
            masks, _ = mask_model(inputs, labels)
            numMask, c, hh, ww = masks.size()
        for i in range(numMask):
            mascara = masks[i].detach().cpu()  #(1, 1, 224, 224)
            print('mascara=', mascara.size())
            # mascara = torch.squeeze(mascara, 0)  #(1, 224, 224)
            mascara = min_max_normalize_tensor(mascara)  #to 0-1
            mascara = mascara.numpy().transpose(1, 2, 0)
            mascara = cv2.resize(mascara, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
            saliency_bboxes, preprocesing_outs, contours, hierarchy = computeBoundingBoxFromMask(mascara)
            bynary = preprocessor.binarize(mascara, thresh=0.1, maxval=1)
            
            show(wait=50, mask=mascara, binary=bynary)

            frames = load_dynamic_image_frames(paths[i])
            
            for fr in frames:
                for s_bbox in saliency_bboxes:
                    cv2.rectangle(fr, (int(s_bbox.pmin.x), int(s_bbox.pmin.y)), (int(s_bbox.pmax.x), int(s_bbox.pmax.y)), constants.yellow, 2)
                cv2.imshow('image', fr)
                if cv2.waitKey(50) & 0xFF == ord('q'):
                    break

if __name__ == "__main__":
    localization()

