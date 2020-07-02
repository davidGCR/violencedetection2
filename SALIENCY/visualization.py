import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import torch
import numpy as np
from VIOLENCE_DETECTION.datasetsMemoryLoader import hockeyLoadData, hockeyTrainTestSplit
from VIOLENCE_DETECTION.transforms import hockeyTransforms
from VIOLENCE_DETECTION.dataloader import MyDataloader
from VIDEO_REPRESENTATION.imageAnalysis import show
from VIDEO_REPRESENTATION.preprocessor import Preprocessor
from SALIENCY.saliencyModel import build_saliency_model
from UTIL.util import min_max_normalize_tensor, min_max_normalize_np
from constants import DEVICE
import constants

def vizualization():
    input_size = 224
    num_classes = 2
    transforms = hockeyTransforms(input_size)
    datasetAll, labelsAll, numFramesAll = hockeyLoadData()
    train_x, train_y, train_numFrames, test_x, test_y, test_numFrames = hockeyTrainTestSplit('train-test-1', datasetAll, labelsAll, numFramesAll)
    default_args = {
            'X': test_x,
            'y': test_y,
            'numFrames': test_numFrames,
            'transform': transforms['val'],
            'NDI': 1,
            'videoSegmentLength': 20,
            'positionSegment': 'begin',
            'overlapping': 0,
            'frameSkip': 0,
            'skipInitialFrames': 0,
            'batchSize': 1,
            'shuffle': False,
            'numWorkers': 4,
            'pptype': None,
            'modelType': None
    }
    dt_loader = MyDataloader(default_args)
    mask_model = build_saliency_model(num_classes)
    mask_model.to(DEVICE)
    file = os.path.join(constants.PATH_RESULTS,
                        'MASKING/checkpoints',
                        'MaskModel_backnone=resnet50_NDI=1_AreaLoss=8_SmoothLoss=0.5_PreservLoss=0.3_AreaLoss2=0.3_epochs=30-epoch-29-loss=2.041108002662659.pth')
    _, foldername = os.path.split(file)
    foldername = os.path.join(constants.PATH_RESULTS, 'MASKING', 'images', foldername[:-4])
    if not os.path.exists(foldername):
        os.mkdir(foldername)

    if DEVICE == 'cuda:0':
        mask_model.load_state_dict(torch.load(file), strict=False)
    else:
        mask_model.load_state_dict(torch.load(file, map_location=DEVICE))
    mask_model.eval()
    preprocessor = Preprocessor(None)
    
    for i, data in enumerate(dt_loader.dataloader):
        inputs, labels, video_names, _ = data
        print('inputs=', i, inputs.size())
        print('Video: ', video_names[0], ', Label: ', labels)
        with torch.no_grad():
            masks, _ = mask_model(inputs, labels)
            mascara = masks.detach().cpu()  #(1, 1, 224, 224)
            
            mascara = torch.squeeze(mascara, 0)  #(1, 224, 224)
            # print('mascara=', mascara.size())
            mascara = min_max_normalize_tensor(mascara)  #to 0-1
            mascara = mascara.numpy().transpose(1, 2, 0)
            bynary = preprocessor.binarize(mascara, thresh=0.1, maxval=1)
            dimages = min_max_normalize_tensor(inputs)
            # batch_size, timesteps, C, H, W = dimages.size()
            # dimages = dimages.view(batch_size * timesteps, C, H, W)
            # dimages = dimages[0]
            dimages = dimages.numpy()[0][0].transpose(1, 2, 0)
            # print('numpy images: ', dimages.shape)
            # dimages = [min_max_normalize_np(np.array(img, dtype="float32")) for img in inputs]
            show(wait=50, dimage=dimages, mask=mascara, binary=bynary)
            if i < 10:
                _, video_name = os.path.split(video_names[0])
                filename = os.path.join(foldername, video_name + '-'+str(labels.item()) + '.jpg')
                # print(filename)
                cv2.imwrite(filename, 255 * mascara)
                filename = os.path.join(foldername, video_name + '-'+str(labels.item()) + '-bin.jpg')
                cv2.imwrite(filename, 255 * bynary)
                
            # cv2.imshow('Mask Image', mascara)
            # if cv2.waitKey(50) & 0xFF == ord('q'):
            #     break


if __name__ == "__main__":
    vizualization()