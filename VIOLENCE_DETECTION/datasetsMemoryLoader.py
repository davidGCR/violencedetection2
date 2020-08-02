import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from UTIL.kfolds import k_folds
from UTIL.util import read_file, save_file, save_csvfile_multicolumn, read_csvfile_threecolumns, sortListByStrNumbers
import constants
import random
import glob
import numpy as np
from operator import itemgetter

from sklearn.model_selection import KFold

def checkBalancedSplit(Y_train, Y_test):
    positive = 0
    posTrain = [1 for y in Y_train if y == 1]
    print('Train-Positives samples={}, Negative samples={}'.format(len(posTrain), len(Y_train) - len(posTrain)))
    posTest = [1 for y in Y_test if y == 1]
    print('Test-Positives samples={}, Negative samples={}'.format(len(posTest), len(Y_test) - len(posTest)))

def customize_kfold(n_splits, dataset, X_len, shuffle=True):
    X=np.arange(X_len)
    if dataset == 'hockey' or dataset == 'ucfcrime2local':
        kfold = KFold(n_splits, shuffle=shuffle)
        folder = constants.PATH_UCFCRIME2LOCAL_README if dataset=='ucfcrime2local' else constants.PATH_HOCKEY_README
        if not os.path.exists(os.path.join(folder, 'fold_1_train.txt')):
            for i, (train_idx, test_idx) in enumerate(kfold.split(X)):
                save_file(train_idx, os.path.join(folder, 'fold_{}_train.txt'.format(i + 1)))
                save_file(test_idx, os.path.join(folder, 'fold_{}_test.txt'.format(i + 1)))
        
        for i in range(n_splits):
            train_idx = read_file(os.path.join(folder, 'fold_{}_train.txt'.format(i + 1)))
            test_idx = read_file(os.path.join(folder, 'fold_{}_test.txt'.format(i + 1)))
            train_idx = list(map(int, train_idx))
            test_idx = list(map(int, test_idx))
            yield train_idx, test_idx
    elif dataset == 'vif':
        splitsLen = []
        folder = os.path.join(constants.PATH_VIF_README)
        if not os.path.exists(os.path.join(folder, 'fold_1_train.txt')):
            if not os.path.exists(os.path.join(folder, 'lengths.txt')):
                for fold in range(n_splits):
                    violence_path = os.path.join(constants.PATH_VIF_FRAMES, str(fold+1),'Violence')
                    non_violence_path = os.path.join(constants.PATH_VIF_FRAMES, str(fold + 1), 'NonViolence')
                    violence_videos = os.listdir(violence_path)
                    non_violence_videos = os.listdir(non_violence_path)
                    splitsLen.append(len(violence_videos) + len(non_violence_videos))
                save_file(splitsLen, os.path.join(folder, 'lengths.txt'))
            else:
                splitsLen = read_file(os.path.join(folder, 'lengths.txt'))
                splitsLen = list(map(int, splitsLen))
            
        for i,l in enumerate(splitsLen):
            end = np.sum(splitsLen[:(i+1)])
            start = end-splitsLen[i]
            test_idx = np.arange(start,end)
            train_idx = np.arange(0, start).tolist() + np.arange(end, len(X)).tolist()
            yield train_idx, test_idx
###################################################################################################################
############################################### Vif #####################################################
###################################################################################################################
def getFoldData(fold_path):
    """
    Load data from folder
    """
    # print('getFoldData: ',fold_path)
    names = []
    labels = []
    num_frames = []

    violence_path = os.path.join(fold_path, 'Violence')
    non_violence_path = os.path.join(fold_path, 'NonViolence')
    violence_videos = os.listdir(violence_path)
    non_violence_videos = os.listdir(non_violence_path)
    for video in violence_videos:
        video_folder = os.path.join(violence_path, video)
        num_frames.append(len(os.listdir(video_folder)))
        names.append(video_folder)
        labels.append(1)
    for video in non_violence_videos:
        video_folder = os.path.join(non_violence_path, video)
        num_frames.append(len(os.listdir(video_folder)))
        names.append(video_folder)
        labels.append(0)

    return names, labels, num_frames

def vifLoadData(folds_dir):
    names, labels, num_frames = [], [], []
    splitsLen = []
    for i in range(5):
        x, y, num_f = getFoldData(os.path.join(folds_dir, str(i + 1)))
        splitsLen.append(len(x))
        # print(x)
        names = names + x
        labels = labels + y
        num_frames = num_frames + num_f
    return  names, labels, num_frames, splitsLen

def train_test_iteration(test_fold_path, shuffle):
    """
        Load Train and test data from pre-determined splits
    """
    test_names, test_labels, test_num_frames = getFoldData(test_fold_path)
    folds_dir, fold_number = os.path.split(test_fold_path)
    fold_number = int(fold_number)
    train_names = []
    train_labels = []
    train_num_frames = []

    for i in range(5):
        if i + 1 != fold_number:
            names, labels, num_frames = getFoldData(os.path.join(folds_dir, str(i + 1)))
            train_names.extend(names)
            train_labels.extend(labels)
            train_num_frames.extend(num_frames)
    if shuffle:
        combined = list(zip(train_names, train_labels, train_num_frames))
        random.shuffle(combined)
        train_names[:], train_labels[:], train_num_frames[:] = zip(*combined)

        combined = list(zip(test_names, test_labels, test_num_frames))
        random.shuffle(combined)
        test_names[:], test_labels[:], test_num_frames[:] = zip(*combined)

    return train_names, train_labels, train_num_frames, test_names, test_labels, test_num_frames

###################################################################################################################
############################################### HOCKEY FIGHTS #####################################################
###################################################################################################################

def hockeyLoadData():
    path_violence = constants.PATH_HOCKEY_FRAMES_VIOLENCE
    path_non_violence = constants.PATH_HOCKEY_FRAMES_NON_VIOLENCE
    if not os.path.exists(os.path.join(constants.PATH_HOCKEY_README, 'all_data_labels_numFrames.csv')):
        datasetAll, labelsAll, numFramesAll = initializeDataset.createDataset(path_violence, path_non_violence, shuffle)  #shuffle
        all_data = zip(datasetAll, labelsAll, numFramesAll)
        save_csvfile_multicolumn(all_data, os.path.join(constants.PATH_HOCKEY_README, 'all_data_labels_numFrames.csv'))
    else:
        datasetAll, labelsAll, numFramesAll = read_csvfile_threecolumns(os.path.join(constants.PATH_HOCKEY_README, 'all_data_labels_numFrames.csv'))
    return datasetAll, labelsAll, numFramesAll

def hockeyTrainTestSplit(split_type, datasetAll, labelsAll, numFramesAll):
    train_idx = read_file(os.path.join(constants.PATH_HOCKEY_README, 'fold_{}_train.txt'.format(int(split_type[len(split_type)-1]))))
    test_idx = read_file(os.path.join(constants.PATH_HOCKEY_README, 'fold_{}_test.txt'.format(int(split_type[len(split_type)-1]))))
    train_idx = list(map(int, train_idx))
    test_idx = list(map(int, test_idx))
    
    train_x = list(itemgetter(*train_idx)(datasetAll))
    train_y = list(itemgetter(*train_idx)(labelsAll))
    train_numFrames = list(itemgetter(*train_idx)(numFramesAll))
    test_x = list(itemgetter(*test_idx)(datasetAll))
    test_y = list(itemgetter(*test_idx)(labelsAll))
    test_numFrames = list(itemgetter(*test_idx)(numFramesAll))

    return train_x, train_y, train_numFrames, test_x, test_y, test_numFrames
    
###################################################################################################################
############################################### UCFCRIME2LOCAL#####################################################
###################################################################################################################

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

def get_Test_Data(fold):
    # train_idx = read_file(os.path.join(constants.PATH_UCFCRIME2LOCAL_README, 'fold-{}-train.txt'.format(fold)))
    test_idx = read_file(os.path.join(constants.PATH_UCFCRIME2LOCAL_README, 'fold-{}-test.txt'.format(fold)))
    # train_idx = list(map(int, train_idx))
    test_idx = list(map(int, test_idx))
    return test_idx

def get_Fold_Data(fold):
    train_idx = read_file(os.path.join(constants.PATH_UCFCRIME2LOCAL_README, 'fold-{}-train.txt'.format(fold)))
    test_idx = read_file(os.path.join(constants.PATH_UCFCRIME2LOCAL_README, 'fold-{}-test.txt'.format(fold)))
    train_idx = list(map(int, train_idx))
    test_idx = list(map(int, test_idx))
    return train_idx, test_idx

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
