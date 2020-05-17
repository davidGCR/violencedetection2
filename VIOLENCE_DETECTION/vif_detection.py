import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import include
import constants

import argparse
import numpy as np

from UTIL.util import video2Images2, sortListByStrNumbers

def preprocessing_dataset(dataset_path_videos, dataset_path_frames):
    folds = os.listdir(dataset_path_videos)
    folds = sortListByStrNumbers(folds)
    for f in folds:
        violence_path = os.path.join(dataset_path_videos, f, 'Violence')
        non_violence_path = os.path.join(dataset_path_videos, f, 'NonViolence')
        violence_videos = os.listdir(violence_path)
        non_violence_videos = os.listdir(non_violence_path)
        # videos = violence_videos + non_violence_videos
        for video in violence_videos:
            video_folder = os.path.join(dataset_path_frames, str(f), 'Violence', video[:-4])
            video_full_path = os.path.join(violence_path,video)
            if not os.path.exists(video_folder):
                os.makedirs(video_folder)
            video2Images2(video_full_path, video_folder)
        
        for video in non_violence_videos:
            video_folder = os.path.join(dataset_path_frames, str(f), 'NonViolence', video[:-4])
            video_full_path = os.path.join(non_violence_path,video)
            if not os.path.exists(video_folder):
                os.makedirs(video_folder)
            video2Images2(video_full_path,video_folder)

def dataset_statistics(dataset_path_frames):
    folds = os.listdir(dataset_path_frames)
    folds = sortListByStrNumbers(folds)
    violence_x = []
    violence_y = []
    violence_num_frames = []
    nonviolence_num_frames = []

    for f in folds:
        violence_path = os.path.join(dataset_path_frames, f, 'Violence')
        non_violence_path = os.path.join(dataset_path_frames, f, 'NonViolence')
        violence_videos = os.listdir(violence_path)
        non_violence_videos = os.listdir(non_violence_path)
        for video in violence_videos:
            video_folder = os.path.join(violence_path, video)
            violence_num_frames.append(len(os.listdir(video_folder)))
            violence_x.append('fold:{}, video: {}, numFrames: {}, label: {}'.format(f, video_folder, len(os.listdir(video_folder)), 1))
        for video in non_violence_videos:
            video_folder = os.path.join(non_violence_path, video)
            nonviolence_num_frames.append(len(os.listdir(video_folder)))
            violence_x.append('fold:{}, video: {}, numFrames: {}, label: {}'.format(f, video_folder, len(os.listdir(video_folder)), 0))
    
    avg_violence_frames = np.average(violence_num_frames)
    avg_nonviolence_frames = np.average(nonviolence_num_frames)

    return violence_num_frames, nonviolence_num_frames, avg_violence_frames, avg_nonviolence_frames, violence_x


def __main__():
    violence_num_frames, nonviolence_num_frames, avg_violence_frames, avg_nonviolence_frames, violence_x = dataset_statistics(constants.PATH_VIF_FRAMES)
    # print(violence_num_frames, 'AVG: ', avg_violence_frames)
    # print(nonviolence_num_frames, 'AVG: ', avg_nonviolence_frames)
    print(violence_x)
    # preprocessing_dataset(constants.PATH_VIF_VIDEOS, constants.PATH_VIF_FRAMES)
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--modelType",type=str,default="alexnet",help="model")
    # parser.add_argument("--numEpochs",type=int,default=30)
    # parser.add_argument("--batchSize",type=int,default=64)
    # parser.add_argument("--featureExtract",type=lambda x: (str(x).lower() == 'true'), default=False, help="to fine tunning")
    # parser.add_argument("--numDynamicImagesPerVideo", type=int)
    # parser.add_argument("--joinType", type=str)
    # parser.add_argument("--foldsNumber", type=int, default=5)
    # parser.add_argument("--numWorkers", type=int, default=4)
    # parser.add_argument("--videoSegmentLength", type=int)
    # parser.add_argument("--positionSegment", type=str)
    # parser.add_argument("--split_type", type=str)
    # parser.add_argument("--overlaping", type=float)
    # parser.add_argument("--frameSkip", type=int, default=0)
    # args = parser.parse_args()

__main__()