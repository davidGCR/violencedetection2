import sys
sys.path.insert(1, '/media/david/datos/PAPERS-SOURCE_CODE/violencedetection')
import os
import constants
import glob
import pandas as pd
import cv2
import numpy as np

def labels_2_binary(multi_labels):
    binary_labels = multi_labels.copy()
    for idx,label in enumerate(multi_labels):
        if label == 0:
            binary_labels[idx]=0
        else:
            binary_labels[idx]=1
    return binary_labels

def train_test_videos(train_file, test_file, g_path):
    """ load train-test split from original dataset """
    train_names = []
    train_labels = []
    test_names = []
    test_labels = []
    train_bbox_files = []
    test_bbox_files = []
    classes = {'Normal_Videos': 0, 'Arrest': 1, 'Assault': 2, 'Burglary': 3, 'Robbery': 4, 'Stealing': 5, 'Vandalism': 6}
    with open(train_file, 'r') as file:
        for row in file:
            train_names.append(os.path.join(g_path,row[:-1]))
            train_labels.append(row[:-4])
            label = row[:-4]
            if label != 'Normal_Videos':
                file = row[:-1] + '.txt'
                file = os.path.join(constants.PATH_UCFCRIME2LOCAL_BBOX_ANNOTATIONS, file)
                train_bbox_files.append(file)
            else:
                train_bbox_files.append(None)

    with open(test_file, 'r') as file:
        for row in file:
            test_names.append(os.path.join(g_path,row[:-1]))
            test_labels.append(row[:-4])
            label = row[:-4]
            if label != 'Normal_Videos':
                file = row[:-1] + '.txt'
                file = os.path.join(constants.PATH_UCFCRIME2LOCAL_BBOX_ANNOTATIONS, file)
                test_bbox_files.append(file)
            else:
                test_bbox_files.append(None)

    # for idx,label in enumerate(train_labels):
    #     if label == 'Normal_Videos':
    #         train_labels[idx]=0
    #     else:
    #         train_labels[idx]=1
    # for idx,label in enumerate(test_labels):
    #     if label == 'Normal_Videos':
    #         test_labels[idx]=0
    #     else:
    #         test_labels[idx]=1       
    train_labels = [classes[label] for label in train_labels]
    test_labels = [classes[label] for label in test_labels]
    # for i in range(len(train_names)):
    #      print(train_names[i])
    NumFrames_train = [len(glob.glob1(train_names[i], "*.jpg")) for i in range(len(train_names))]
    NumFrames_test = [len(glob.glob1(test_names[i], "*.jpg")) for i in range(len(test_names))]
    return train_names, train_labels, NumFrames_train, train_bbox_files, test_names, test_labels, NumFrames_test, test_bbox_files

def only_anomaly_test_videos(test_file, g_path):
    """ load train-test split from original dataset """
    test_names = []
    test_labels = []
    test_bbox_files = []
    classes = {'Normal_Videos': 0, 'Arrest': 1, 'Assault': 2, 'Burglary': 3, 'Robbery': 4, 'Stealing': 5, 'Vandalism': 6}

    with open(test_file, 'r') as file:
        for row in file:
            label = row[:-4]
            if label != 'Normal_Videos':
                file = row[:-1] + '.txt'
                file = os.path.join(constants.PATH_UCFCRIME2LOCAL_BBOX_ANNOTATIONS, file)
                test_bbox_files.append(file)
                # print('--> file: ',file)
                test_names.append(os.path.join(g_path,row[:-1]))
                test_labels.append(label)     
    test_labels = [classes[label] for label in test_labels]
    NumFrames_test = [len(glob.glob1(test_names[i], "*.jpg")) for i in range(len(test_names))]
    return test_names, test_labels, NumFrames_test, test_bbox_files



## process the Temporal_Anomaly_Annotation_for_Testing_Videos.txt        
def cutVideo(path):
    data = pd.read_csv(path, sep='  ') #name anomaly  start1  end1  start2  end2
    print(data.head())
    videos = data["name"].values
    anomaly = data["anomaly"].values
    start1 = data["start1"].values
    end1 = data["end1"].values
    start2 = data["start2"].values
    end2 = data["end2"].values
    # videos = [video.split("_")[0] for video in videos]
    print(videos)
    print(len(videos))
    
    return videos, anomaly, start1,end1, start2, end2


def normalVideoNormalize(num_avg_frames, video):
    path_frames_out = os.path.join(constants.PATH_UCFCRIME2LOCAL_FRAMES_REDUCED,video[:-9])#folder
    if not os.path.exists(path_frames_out):
        os.makedirs(path_frames_out)
    vid = cv2.VideoCapture(os.path.join(constants.PATH_UCFCRIME2LOCAL_VIDEOS,video))
    index_frame = 0
    while(True):
        ret, frame = vid.read()
        if not ret:
            print('Houston there is a problem...')
            break
        index_frame += 1
        if index_frame < num_avg_frames:
            
            cv2.imwrite(os.path.join(path_frames_out, 'frame' + str("{0:03}".format(index_frame)) + '.jpg'), frame)

def createReducedDataset():
    """Create videos with only frames annotated with bounding box"""
    # videos = os.listdir(constants.PATH_UCFCRIME2LOCAL_VIDEOS)
    lista_videos = os.listdir(constants.PATH_UCFCRIME2LOCAL_VIDEOS)
    lista_videos.sort()
    NUM_AVG_FRAMES = 385

    for video in lista_videos:
        if video[0:6] == 'Normal':
            normalVideoNormalize(NUM_AVG_FRAMES,video)
            continue
        # print(video)
        bdx_file_path = os.path.join(constants.PATH_UCFCRIME2LOCAL_BBOX_ANNOTATIONS, video[:-9]+'.txt')
        # print(bdx_file_path)
        path_frames_out = os.path.join(constants.PATH_UCFCRIME2LOCAL_FRAMES_REDUCED,video[:-9])
        if not os.path.exists(path_frames_out):
            os.makedirs(path_frames_out)
        data = []
        with open(bdx_file_path, 'r') as file:
            for row in file:
                data.append(row.split())
        data = np.array(data)
        vid = cv2.VideoCapture(os.path.join(constants.PATH_UCFCRIME2LOCAL_VIDEOS,video))
        index_frame = 0
        frames_numbers = []
        
        while(True):
            ret, frame = vid.read()
            if not ret:
                # print('Houston there is a problem...')
                break
            index_frame += 1
            if index_frame < data.shape[0]:
                if int(data[index_frame, 6]) == 0:
                    frames_numbers.append(index_frame)
                    # cv2.imwrite(os.path.join(path_frames_out, 'frame' + str("{0:03}".format(index_frame)) + '.jpg'), frame)
                    # frame = cv2.rectangle(frame,(int(data[index_frame,1]),int(data[index_frame,2])),(int(data[index_frame,3]),int(data[index_frame,4])),(0,255,0))
        
        print('-'*20,video, len(frames_numbers))
        for idx in range(len(frames_numbers)):
            if (idx + 1) < len(frames_numbers):
                # print(frames_numbers[idx + 1], frames_numbers[idx])
                if int(frames_numbers[idx + 1]) - int(frames_numbers[idx]) < 4:
                    vid.set(cv2.CAP_PROP_POS_FRAMES, int(frames_numbers[idx]))
                    ret, frame = vid.read()
                    if not ret:
                        print('Houston there is a problem...')
                        break
                    cv2.imwrite(os.path.join(path_frames_out, 'frame' + str("{0:03}".format(frames_numbers[idx])) + '.jpg'), frame)
                    # print(os.path.join(path_frames_out, 'frame' + str("{0:03}".format(frames_numbers[idx])) + '.jpg'))
                else:
                    break
        # print('frames for videos: ', len(frames_numbers))
    
def numFramesMean(path=constants.PATH_UCFCRIME2LOCAL_FRAMES_REDUCED):
    lista_videos_folders = os.listdir(path)
    lista_videos_folders.sort()
    total_videos = len(lista_videos_folders)
    total = 0
    for i, vid_folder in enumerate(lista_videos_folders):
        numFrames = len(glob.glob1(os.path.join(path, vid_folder), "*.jpg"))
        total += numFrames
        print(os.path.join(path, vid_folder), numFrames)
    avg = int(total / total_videos) #385
    print(avg)

def extractMetadata(path='/media/david/datos/Violence DATA/AnomalyCRIME/UCFCrime2Local/videos'):
    paths = os.listdir(path)
    paths.sort()
    # r = re.compile("([a-zA-Z]+)([0-9]+)")
    # labels = [r.match(string).groups() for string in paths] # (Robbery,089)
    # names = [str(tup[0])+str(tup[1]) for tup in labels] #Robbery089
    # labels = [tup[0] for tup in labels]  #Robbery
    names = [string[:-9] for string in paths]
    labels = [string[:-12] for string in paths]
    return names, labels, paths

def videos2frames(path_videos, path_frames):
#   listViolence = os.listdir(path_videos)
#   listViolence.sort()
    names, _, paths = extractMetadata(path_videos)
    # print(paths)
    # print(names)
    for idx,video in enumerate(paths):
        path_video = os.path.join(path_videos, video) #/media/david/datos/Violence DATA/AnomalyCRIME/UCFCrime2Local/videos/Vandalism050_x264.mp4
        # print('in: ',path_video)
        # path_frames_out = os.path.join(path_frames, str(idx + 1)) #/media/david/datos/Violence DATA/AnomalyCRIME/UCFCrime2Local/frames/violence/100
        path_frames_out = os.path.join(path_frames, names[idx])
        # print(path_frames_out)
        if not os.path.exists(path_frames_out):
            os.makedirs(path_frames_out)
        dirContents = os.listdir(path_frames_out)
        if len(dirContents) == 0:
            video2Images2(path_video, path_frames_out)