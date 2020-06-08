import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dynamicImage import getDynamicImage
from VIOLENCE_DETECTION.datasetsPreprocessing import hockeyLoadData, hockeyTrainTestSplit, crime2localLoadData
from VIOLENCE_DETECTION.dataloader import Dataloader
from VIOLENCE_DETECTION.transforms import hockeyTransforms
from UTIL.util import min_max_normalize_np, min_max_normalize_tensor
import cv2
import numpy as np
from PIL import Image
import constants
import matplotlib

def show(wait, **kwargs):
    pos_x = 20
    sep = 400
    # while(True):
    for idx, key in enumerate(kwargs):
        print('key=',key, 'value=',type(kwargs[key]))
        cv2.imshow(key, kwargs[key])
        cv2.namedWindow(key)#x,y
        cv2.moveWindow(key, pos_x+idx*sep, 100)
        if cv2.waitKey(wait) & 0xFF == ord('q'):
            break

def main():
    split_type = 'train-test-1'
    datasetAll, labelsAll, numFramesAll = hockeyLoadData()
    train_x, train_y, train_numFrames, test_x, test_y, test_numFrames = hockeyTrainTestSplit(split_type, datasetAll, labelsAll, numFramesAll)
    numDynamicImagesPerVideo = 1
    positionSegment = 'begin'
    # videoSegmentLength = 40
    overlapping = 0
    frameSkip = 0
    skipInitialFrames = False
    segmentPreprocessing = False
    batchSize = 8
    numWorkers = 4
    transforms = hockeyTransforms(224)
    # train_x, train_y, train_numFrames, test_x, test_y, test_numFrames = hockeyTrainTestSplit(split_type, datasetAll, labelsAll, numFramesAll)
    hockey_dt_loader = Dataloader(X=datasetAll,
                                y=labelsAll,
                                numFrames=numFramesAll,
                                transform=transforms['train'],
                                NDI=numDynamicImagesPerVideo,
                                videoSegmentLength=None,
                                positionSegment=positionSegment,
                                overlapping=overlapping,
                                frameSkip=frameSkip,
                                skipInitialFrames=skipInitialFrames,
                                segmentPreprocessing=segmentPreprocessing,
                                batchSize=batchSize,
                                shuffle=True,
                                numWorkers=numWorkers)
    
    X, y, numFrames = crime2localLoadData(min_frames=40)
    skipInitialFrames = 10
    ucf_dt_loader = Dataloader(X=X,
                                y=y,
                                numFrames=numFrames,
                                transform=None,
                                NDI=1,
                                videoSegmentLength=None,
                                positionSegment=positionSegment,
                                overlapping=overlapping,
                                frameSkip=frameSkip,
                                skipInitialFrames=skipInitialFrames,
                                segmentPreprocessing=segmentPreprocessing,
                                batchSize=batchSize,
                                shuffle=True,
                                numWorkers=numWorkers)

    video = '1'
    label = 1
    transform = False
    preprocess = False
    lens = [2,5,10,15,20,25,30,35,40]
    for segmentLen in lens:
        dimagPath = os.path.join(constants.PATH_RESULTS,'dimages','vd_{}_class({})_len({})_pp({}).png'.format(video, label, segmentLen, preprocess))
        frames, dimages, dimages2, label = hockey_dt_loader.getElement(name=video, label=label, savePath=dimagPath, seqLen=segmentLen, transform=transform, preprocess=preprocess)
        dimage = min_max_normalize_np(np.array(dimages[0]))
        # dimage2 = min_max_normalize_tensor(dimages2)
        # dimage2 = dimage2.numpy()[0].transpose(1, 2, 0)
        frame = frames[0][0]
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        show(wait=1000,video_frame=frame, dimage=dimage)

if __name__ == "__main__":
    main()