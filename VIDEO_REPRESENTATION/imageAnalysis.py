import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dynamicImage import getDynamicImage
from VIOLENCE_DETECTION.datasetsMemoryLoader import hockeyLoadData, hockeyTrainTestSplit, crime2localLoadData, vifLoadData
from VIOLENCE_DETECTION.dataloader import MyDataloader
from VIOLENCE_DETECTION.transforms import hockeyTransforms, vifTransforms
from UTIL.util import min_max_normalize_np, min_max_normalize_tensor
from VIDEO_REPRESENTATION.preprocessor import Preprocessor
import cv2
import numpy as np
from PIL import Image
import constants
import matplotlib

import tkinter

# def moveWindow(id,):

def show(wait, **kwargs):
    root = tkinter.Tk()
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()
    # print('Screen: ', width, height)
    pos_x = 20
    pos_y = 100
    w = 360
    h = 288
    xsep = w+20
    ysep = h+20
    # max_cols = int(width / w - 1)
    max_cols = 5
    numImages = len(kwargs.items())
    row=0

    for idx, key in enumerate(kwargs):
        if kwargs[key] is None:
            break
        row = idx//max_cols
        if idx+1 > max_cols:
            posx = pos_x + (idx - max_cols) * xsep
        else:
            posx = pos_x + idx * xsep
        posy = pos_y + row * ysep
        
        if key == 'video_frames':
            for i, frame in enumerate(kwargs[key]):
                n = 'frame'
                # print('--frame={}'.format(i+1))
                cv2.imshow(n, frame)
                cv2.namedWindow(n)  #x,y
                cv2.moveWindow(n, posx, posy)
        else:
            cv2.imshow(key, kwargs[key])
            cv2.namedWindow(key)#x,y
            cv2.moveWindow(key, posx, posy)
        # wsize = cv2.getWindowImageRect('frame')
        # print(wsize)
        if cv2.waitKey(wait) & 0xFF == ord('q'):
            break
def hockey():
    split_type = 'train-test-1'
    datasetAll, labelsAll, numFramesAll = hockeyLoadData()
    # train_x, train_y, train_numFrames, test_x, test_y, test_numFrames = hockeyTrainTestSplit(split_type, datasetAll, labelsAll, numFramesAll)
    numDynamicImagesPerVideo = 1
    positionSegment = 'begin'
    # videoSegmentLength = 40
    overlapping = 0
    frameSkip = 0
    skipInitialFrames = False
    ptype = 'ok'
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
                                batchSize=batchSize,
                                shuffle=True,
                                numWorkers=numWorkers,
                                pptype=ptype)
    
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
                                pptype=ptype,
                                batchSize=batchSize,
                                shuffle=True,
                                numWorkers=numWorkers)

    video = 'Normal_Videos-Arrest003-NSplit-1'
    label = 0
    transform = False
    ptype = None
    lens = [2, 5, 10, 15, 20, 25, 30, 35, 40]
    # lens = [2, 5, 10]
    myPreprocesor = Preprocessor(pType=None)
    ndi = 3
    
    for segmentLen in lens:
        dimagPath = os.path.join(constants.PATH_RESULTS,'dimages','vd_{}_class({})_len({})_pp({}).png'.format(video, label, segmentLen, ptype))
        frames, dimages, dimages2, label = ucf_dt_loader.getElement(name=video, label=label, savePath=None, ndi=ndi, seqLen=segmentLen, transform=transform, ptype=ptype)
        dimages = [min_max_normalize_np(np.array(img, dtype="float32")) for img in dimages]
        # dimage2 = min_max_normalize_tensor(dimages2)
        # dimage2 = dimage2.numpy()[0].transpose(1, 2, 0)
        # frame = frames[0][0]
        # cv2.imshow('ok', dimages[1])
        # if cv2.waitKey(1000) & 0xFF == ord('q'):
        #     break
        
        for i,frame in enumerate(frames[0]):
            frames[0][i] = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        previous = frames[0][0]
        current = frames[0][len(frames[0]) - 1]
        # previous = dimages[0]
        # current = dimages[len(dimages) - 1]

        print('Format dimage: ',type(dimages[0]), dimages[0].dtype)

        diff1, diff2, thresh_binary, thresh_otsu = myPreprocesor.bakgroundFrameDifference(current=current, previous=previous)
        show(wait=800,video_frames=frames[0], dimage=dimages[0], previous=previous, current=current, diff1=diff1, diff2=diff2, thresh_binary=thresh_binary, thresh_otsu=thresh_otsu)

def main():
    # split_type = 'train-test-1'
    datasetAll, labelsAll, numFramesAll, _ = vifLoadData(constants.PATH_VIF_FRAMES)
    # print(datasetAll[0:10])
    args = {
        'X': datasetAll,
        'y': labelsAll,
        'numFrames': numFramesAll,
        'transform': None,
        'NDI': 1,
        'videoSegmentLength': 20,
        'positionSegment': 'begin',
        'overlapping': 0,
        'frameSkip': 0,
        'skipInitialFrames': 0,
        'batchSize': 8,
        'shuffle': False,
        'numWorkers': 4,
        'pptype': None, 
    }
    
    dt_loader = MyDataloader(args)
    video = 'football_crowds__HARRY_KEWELL_SCORES_CROWD_GO_NUTS__phillipstama__K7qlpU7tMhQ'
    index, v_name = dt_loader.dataset.getindex(video)
    # print(type(index))
    print('index={}, name={}, label={}, numFrames={}'.format(index, datasetAll[index], labelsAll[index], type(numFramesAll[index])))
    lens = [2, 5, 10, 15, 20, 25, 30]
    # lens = [2, 5, 10]
    myPreprocesor = Preprocessor(pType=None)
    for segmentLen in lens:
        dimagPath = os.path.join(constants.PATH_RESULTS, 'dimages', 'vd_{}_class({})_len({})_pp({}).png'.format(datasetAll[index], labelsAll[index], segmentLen, args['pptype']))
        print('*'*10, segmentLen)
        frames, dimages, dimages2, label = dt_loader.dataset.getOneItem(int(index), transform=False, ptype=args['pptype'], savePath=None, ndi=args['NDI'], seqLen=segmentLen)
        dimages = [min_max_normalize_np(np.array(img, dtype="float32")) for img in dimages]
        # dimage2 = min_max_normalize_tensor(dimages2)
        # dimage2 = dimage2.numpy()[0].transpose(1, 2, 0)
        # frame = frames[0][0]
        # cv2.imshow('ok', dimages[1])
        # if cv2.waitKey(1000) & 0xFF == ord('q'):
        #     break
        
        for i,frame in enumerate(frames[0]):
            frames[0][i] = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        previous = frames[0][0]
        current = frames[0][len(frames[0]) - 1]
        # previous = dimages[0]
        # current = dimages[len(dimages) - 1]
        # print('Format dimage: ',type(dimages[0]), dimages[0].dtype)

        diff1, diff2, thresh_binary, thresh_otsu = myPreprocesor.bakgroundFrameDifference(current=current, previous=previous)
        show(wait=800,video_frames=frames[0], dimage=dimages[0], previous=previous, current=current)

if __name__ == "__main__":
    main()