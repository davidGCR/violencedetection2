import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from UTIL.kfolds import k_folds
from UTIL.util import read_file, save_file, save_csvfile_multicolumn, read_csvfile_threecolumns, sortListByStrNumbers
import constants
import random
import glob
import numpy as np

from sklearn.model_selection import KFold


def crime2localLoadData(min_frames):
    if not os.path.exists(os.path.join(constants.PATH_UCFCRIME2LOCAL_README, 'All_data.txt')):
        train_violence = os.path.join(constants.PATH_UCFCRIME2LOCAL_README, 'Train_violence_split.txt')
        train_nonviolence = os.path.join(constants.PATH_UCFCRIME2LOCAL_README, 'Train_nonviolence_split.txt')
        test_violence = os.path.join(constants.PATH_UCFCRIME2LOCAL_README, 'Test_violence_split.txt')
        test_nonviolence = os.path.join(constants.PATH_UCFCRIME2LOCAL_README, 'Test_nonviolence_split.txt')

        X_train_pos = read_file(train_violence)
        y_train_pos  = [1 for i in range(len(X_train_pos))]
        X_test_pos = read_file(test_violence)
        y_test_pos  = [1 for i in range(len(X_test_pos))]

        X_train_neg = read_file(train_nonviolence)
        sampled_list = random.sample(X_train_neg, len(X_train_pos))
        X_train_neg = sampled_list
        y_train_neg  = [0 for i in range(len(X_train_neg))]
        
        X_test_neg = read_file(test_nonviolence)
        sampled_list = random.sample(X_test_neg, len(X_test_pos))
        X_test_neg = sampled_list
        y_test_neg  = [0 for i in range(len(X_test_neg))]

        X_train_pos = [os.path.join(constants.PATH_UCFCRIME2LOCAL_FRAMES_VIOLENCE, X_train_pos[i]) for i in range(len(X_train_pos))]
        X_train_neg = [os.path.join(constants.PATH_UCFCRIME2LOCAL_FRAMES_NONVIOLENCE, X_train_neg[i]) for i in range(len(X_train_neg))]
        X_test_pos = [os.path.join(constants.PATH_UCFCRIME2LOCAL_FRAMES_VIOLENCE, X_test_pos[i]) for i in range(len(X_test_pos))]
        X_test_neg = [os.path.join(constants.PATH_UCFCRIME2LOCAL_FRAMES_NONVIOLENCE, X_test_neg[i]) for i in range(len(X_test_neg))]

        y = y_train_pos + y_train_neg + y_test_pos + y_test_neg
        X = X_train_pos + X_train_neg + X_test_pos + X_test_neg
        numFrames = [len(glob.glob1(X[i], "*.jpg")) for i in range(len(X))]
        
        yxn = list(zip(y, X, numFrames))
        yxn_sorted = [(y, x, n) for y, x, n in yxn if n > min_frames]
        y, X, numFrames = zip(*yxn_sorted)

        X_t = []
        for i in range(len(X)):
            _, vname = os.path.split(X[i])
            X_t.append(vname)
        X = X_t    
        save_csvfile_multicolumn(zip(X, y, numFrames), os.path.join(constants.PATH_UCFCRIME2LOCAL_README, 'All_data.txt'))
    else:
        X, y, numFrames = read_csvfile_threecolumns(os.path.join(constants.PATH_UCFCRIME2LOCAL_README, 'All_data.txt'))
        X_t = []
        for i in range(len(X)):
            if y[i] == 1:
                X[i] = os.path.join(constants.PATH_UCFCRIME2LOCAL_FRAMES_VIOLENCE, X[i])
            else:
                X[i] = os.path.join(constants.PATH_UCFCRIME2LOCAL_FRAMES_NONVIOLENCE, X[i])


    return X, y, numFrames

def crime2localgGetSplit(X, y, numFrames, splits=5):
    # print(X)
    kfold = KFold(splits, shuffle=True)
    
    if not os.path.exists(os.path.join(constants.PATH_UCFCRIME2LOCAL_README, 'fold-1-train.txt')):
        for i, (train_idx, test_idx) in enumerate(kfold.split(X)):
            save_file(train_idx, os.path.join(constants.PATH_UCFCRIME2LOCAL_README, 'fold-{}-train.txt'.format(i + 1)))
            save_file(test_idx, os.path.join(constants.PATH_UCFCRIME2LOCAL_README, 'fold-{}-test.txt'.format(i + 1)))
    
    for i in range(splits):
        train_idx = read_file(os.path.join(constants.PATH_UCFCRIME2LOCAL_README, 'fold-{}-train.txt'.format(i + 1)))
        test_idx = read_file(os.path.join(constants.PATH_UCFCRIME2LOCAL_README, 'fold-{}-test.txt'.format(i + 1)))
        train_idx = list(map(int, train_idx))
        test_idx = list(map(int, test_idx))
        yield train_idx, test_idx

def getBBoxLabels(video):
    video_folder, video_name = os.path.split(video)
    video_name = video_name[:-8]
    Txt_file = os.path.join(constants.PATH_UCFCRIME2LOCAL_Txt_ANNOTATIONS, video_name)
    bbox_file = os.path.join(constants.PATH_UCFCRIME2LOCAL_BBOX_ANNOTATIONS, video_name)

    frame_list = os.listdir(video)
    frame_list = sortListByStrNumbers(frame_list)
    frame_begin = frame_list[0]
    frame_begin = int(frame_begin[5:-4])
    frame_end = frame_list[len(frame_list) - 1]
    frame_end = int(frame_end[5:-4])

    print('Begin={}, End={}'.format(frame_begin, frame_end))
    data=[]
    with open(Txt_file+'.txt', 'r') as file:
        for row in file:
            data.append(row.split())
    data = np.array(data)

    print(data[11])
    bbox = []
    for row in data:
        num_frame = row[6]
        
        flac = row[7]
        print(num_frame)
    



if __name__ == "__main__":
    # crime2localCreateSplits()
    getBBoxFile('/Users/davidchoqueluqueroman/Desktop/PAPERS-CODIGOS/violencedetection2/DATASETS/CrimeViolence2LocalDATASET/frames/violence/Arrest003-VSplit1')
