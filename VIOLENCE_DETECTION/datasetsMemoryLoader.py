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
import cv2
from sklearn.model_selection import KFold, StratifiedKFold
import re
import shutil
from sklearn.model_selection import train_test_split

def checkBalancedSplit(Y_train, Y_test):
    positive = 0
    posTrain = [1 for y in Y_train if y == 1]
    print('Train-Positives samples={}, Negative samples={}'.format(len(posTrain), len(Y_train) - len(posTrain)))
    posTest = [1 for y in Y_test if y == 1]
    print('Test-Positives samples={}, Negative samples={}'.format(len(posTest), len(Y_test) - len(posTest)))

def load_fold_data(dataset, fold):
    if dataset == 'hockey' or dataset == 'ucfcrime2local' or dataset == 'vif':
        if dataset == 'vif':
            folder = constants.PATH_VIF_README
        else:
            folder = constants.PATH_UCFCRIME2LOCAL_README if dataset=='ucfcrime2local' else constants.PATH_HOCKEY_README
        train_idx = read_file(os.path.join(folder, 'fold_{}_train.txt'.format(fold)))
        test_idx = read_file(os.path.join(folder, 'fold_{}_test.txt'.format(fold)))
        train_idx = list(map(int, train_idx))
        test_idx = list(map(int, test_idx))
        return train_idx, test_idx

# def customize_kfold(n_splits, dataset, X, y, shuffle=True):
def customize_kfold(n_splits, dataset, shuffle=True):
    # X=np.arange(X_len)
    if dataset == 'hockey' or dataset == 'ucfcrime2local':
        kfold = StratifiedKFold(n_splits, shuffle=shuffle)
        folder = constants.PATH_UCFCRIME2LOCAL_README if dataset=='ucfcrime2local' else constants.PATH_HOCKEY_README
        if not os.path.exists(os.path.join(folder, 'fold_1_train.txt')):
            # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            for i, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
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
                if shuffle:
                    random.shuffle(train_idx)
                    random.shuffle(test_idx)
                save_file(train_idx, os.path.join(folder, 'fold_{}_train.txt'.format(i + 1)))
                save_file(test_idx, os.path.join(folder, 'fold_{}_test.txt'.format(i + 1)))
        for i in range(n_splits):
            train_idx = read_file(os.path.join(folder, 'fold_{}_train.txt'.format(i + 1)))
            test_idx = read_file(os.path.join(folder, 'fold_{}_test.txt'.format(i + 1)))
            train_idx = list(map(int, train_idx))
            test_idx = list(map(int, test_idx))
            yield train_idx, test_idx
    elif dataset == 'rwf-2000':
        train_idx, test_idx = [], []
        for i in range(1):
            yield train_idx, test_idx
###################################################################################################################
############################################### RWF-2000 ##########################################################
###################################################################################################################
def rwf_readVideo_saveFrames(file_path, frames_folder_path, resize=(224,224)):
    # Load video
    cap = cv2.VideoCapture(file_path)
    success, frame = cap.read()
    count = 1

    while success:
        filename = os.path.join(frames_folder_path, "frame%d.jpg" % count)
        frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
        cv2.imwrite(filename, frame)     # save frame as JPEG file
        success, frame = cap.read()
        count += 1
    cap.release()

def rwf_videos2frames():
    videos_train_path = os.path.join(constants.PATH_RWF_2000_VIDEOS, 'train')
    videos_test_path = os.path.join(constants.PATH_RWF_2000_VIDEOS,'val')

    frames_train_path = os.path.join(constants.PATH_RWF_2000_FRAMES, 'train')
    frames_test_path = os.path.join(constants.PATH_RWF_2000_FRAMES,'val')

    all_dataset_videos = [videos_train_path, videos_test_path]
    all_dataset_frames = [frames_train_path, frames_test_path]
    for i, split in enumerate(all_dataset_videos):
        for classs in os.listdir(split):
            k=1
            for video in os.listdir(os.path.join(split, classs)):
                video_name = video[:-4]
                vd_pth = os.path.join(split, classs, video)
                frames_folder = os.path.join(all_dataset_frames[i], classs, video_name)
                print(k, '\t', frames_folder)
                # print(k, '\t', vd_pth)
                k += 1
                if not os.path.exists(frames_folder):
                    os.makedirs(frames_folder)
                    rwf_readVideo_saveFrames(vd_pth, frames_folder)
                else:
                    print('Done!!')

def rwf_load_data(shuffle=True, save_csv=False):
    train_names = []
    train_labels = []
    train_num_frames = []
    test_names = []
    test_labels = []
    test_num_frames = []

    train_path = os.path.join(constants.PATH_RWF_2000_FRAMES, 'train')

    for clase in os.listdir(train_path):
        for vid_folder in os.listdir(os.path.join(train_path,clase)):
            sample_path = os.path.join(train_path, clase, vid_folder)
            if os.path.isdir(sample_path):
                train_names.append(sample_path)
                if clase == 'Fight':
                    train_labels.append(1)
                else:
                    train_labels.append(0)
                train_num_frames.append(len(os.listdir(sample_path)))

    test_path = os.path.join(constants.PATH_RWF_2000_FRAMES, 'val')

    for clase in os.listdir(test_path):
        for vid_folder in os.listdir(os.path.join(test_path,clase)):
            sample_path = os.path.join(test_path, clase, vid_folder)
            if os.path.isdir(sample_path):
                test_names.append(sample_path)
                if clase == 'Fight':
                    test_labels.append(1)
                else:
                    test_labels.append(0)
                test_num_frames.append(len(os.listdir(sample_path)))

    if shuffle:
        combined = list(zip(train_names, train_labels, train_num_frames))
        random.shuffle(combined)
        train_names[:], train_labels[:], train_num_frames[:] = zip(*combined)
        combined = list(zip(test_names, test_labels, test_num_frames))
        random.shuffle(combined)
        test_names[:], test_labels[:], test_num_frames[:] = zip(*combined)

    if save_csv:
        all_data_train = zip(train_names, train_labels, train_num_frames)
        save_csvfile_multicolumn(all_data_train, os.path.join(constants.PATH_RWF_2000_README, 'all_data_labels_numFrames_train.csv'))
        all_data_test = zip(test_names, test_labels, test_num_frames)
        save_csvfile_multicolumn(all_data_test, os.path.join(constants.PATH_RWF_2000_README, 'all_data_labels_numFrames_test.csv'))

    return train_names, train_labels, train_num_frames, test_names, test_labels, test_num_frames



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

# def loadHockeyData(path_violence, path_non_violence, shuffle):
#     datasetAll, labelsAll, numFramesAll = [], [], []
#     v_videos = os.listdir(path_violence)
#     v_videos = sortListByStrNumbers(v_videos)
#     nv_videos = os.lisdir(path_non_violence)
#     nv_videos = sortListByStrNumbers(nv_videos)

#     for vd in v_videos:
#         d = os.path.join(path_violence, vd)
#         if not os.path.isdir(d):
#             continue
#         datasetAll.append(d)
#         labelsAll.append(1)
#     # imagesNoF = []
#     for vd in nv_videos:
#         d = os.path.join(path_non_violence, vd)
#         if not os.path.isdir(d):
#             continue
#         datasetAll.append(d)
#         labelsAll.append(0)
#     # Dataset = imagesF + imagesNoF
#     # Labels = list([1] * len(imagesF)) + list([0] * len(imagesNoF))
#     numFramesAll = [len(glob.glob1(datasetAll[i], "*.jpg")) for i in range(len(datasetAll))]
#     return datasetAll, labelsAll, numFramesAll


def hockeyLoadData(shuffle=True):
    path_violence = constants.PATH_HOCKEY_FRAMES_VIOLENCE
    path_non_violence = constants.PATH_HOCKEY_FRAMES_NON_VIOLENCE
    # if not os.path.exists(os.path.join(constants.PATH_HOCKEY_README, 'all_data_labels_numFrames.csv')):
    #     datasetAll, labelsAll, numFramesAll = initializeDataset.createDataset(path_violence, path_non_violence, shuffle)  #shuffle
    #     all_data = zip(datasetAll, labelsAll, numFramesAll)
    #     save_csvfile_multicolumn(all_data, os.path.join(constants.PATH_HOCKEY_README, 'all_data_labels_numFrames.csv'))
    # else:
    #     datasetAll, labelsAll, numFramesAll = read_csvfile_threecolumns(os.path.join(constants.PATH_HOCKEY_README, 'all_data_labels_numFrames.csv'))
    # return datasetAll, labelsAll, numFramesAll
    if not os.path.exists(os.path.join(constants.PATH_HOCKEY_README, 'all_data_labels_numFrames.csv')):
        datasetAll, labelsAll, numFramesAll = [], [], []
        v_videos = os.listdir(path_violence)
        v_videos = sortListByStrNumbers(v_videos)
        nv_videos = os.listdir(path_non_violence)
        nv_videos = sortListByStrNumbers(nv_videos)

        for vd in v_videos:
            d = os.path.join(path_violence, vd)
            if not os.path.isdir(d):
                continue
            datasetAll.append(d)
            labelsAll.append(1)
        # imagesNoF = []
        for vd in nv_videos:
            d = os.path.join(path_non_violence, vd)
            if not os.path.isdir(d):
                continue
            datasetAll.append(d)
            labelsAll.append(0)
        # Dataset = imagesF + imagesNoF
        # Labels = list([1] * len(imagesF)) + list([0] * len(imagesNoF))
        numFramesAll = [len(glob.glob1(datasetAll[i], "*.jpg")) for i in range(len(datasetAll))]
        if shuffle:
            combined = list(zip(datasetAll, labelsAll, numFramesAll))
            random.shuffle(combined)
            datasetAll[:], labelsAll[:], numFramesAll[:] = zip(*combined)

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
        X, y, numFrames = rebuild_load_data()
        for i in range(len(X)):
            if y[i]==1:
                X[i]=os.path.join(constants.PATH_UCFCRIME2LOCAL_FRAMES_VIOLENCE, X[i])
            else:
                X[i]=os.path.join(constants.PATH_UCFCRIME2LOCAL_FRAMES_NONVIOLENCE, X[i])
        # train_violence = os.path.join(constants.PATH_UCFCRIME2LOCAL_README, 'Train_violence_split.txt')
        # train_nonviolence = os.path.join(constants.PATH_UCFCRIME2LOCAL_README, 'Train_nonviolence_split.txt')
        # test_violence = os.path.join(constants.PATH_UCFCRIME2LOCAL_README, 'Test_violence_split.txt')
        # test_nonviolence = os.path.join(constants.PATH_UCFCRIME2LOCAL_README, 'Test_nonviolence_split.txt')

        # X_train_pos = read_file(train_violence)
        # y_train_pos  = [1 for i in range(len(X_train_pos))]
        # X_test_pos = read_file(test_violence)
        # y_test_pos  = [1 for i in range(len(X_test_pos))]

        # X_train_neg = read_file(train_nonviolence)
        # sampled_list = random.sample(X_train_neg, len(X_train_pos))
        # X_train_neg = sampled_list
        # y_train_neg  = [0 for i in range(len(X_train_neg))]

        # X_test_neg = read_file(test_nonviolence)
        # sampled_list = random.sample(X_test_neg, len(X_test_pos))
        # X_test_neg = sampled_list
        # y_test_neg  = [0 for i in range(len(X_test_neg))]

        # X_train_pos = [os.path.join(constants.PATH_UCFCRIME2LOCAL_FRAMES_VIOLENCE, X_train_pos[i]) for i in range(len(X_train_pos))]
        # X_train_neg = [os.path.join(constants.PATH_UCFCRIME2LOCAL_FRAMES_NONVIOLENCE, X_train_neg[i]) for i in range(len(X_train_neg))]
        # X_test_pos = [os.path.join(constants.PATH_UCFCRIME2LOCAL_FRAMES_VIOLENCE, X_test_pos[i]) for i in range(len(X_test_pos))]
        # X_test_neg = [os.path.join(constants.PATH_UCFCRIME2LOCAL_FRAMES_NONVIOLENCE, X_test_neg[i]) for i in range(len(X_test_neg))]

        # y = y_train_pos + y_train_neg + y_test_pos + y_test_neg
        # X = X_train_pos + X_train_neg + X_test_pos + X_test_neg
        # numFrames = [len(glob.glob1(X[i], "*.jpg")) for i in range(len(X))]

        # yxn = list(zip(y, X, numFrames))
        # yxn_sorted = [(y, x, n) for y, x, n in yxn if n > min_frames]
        # y, X, numFrames = zip(*yxn_sorted)

        # X_t = []
        # for i in range(len(X)):
        #     _, vname = os.path.split(X[i])
        #     X_t.append(vname)
        # X = X_t
        # save_csvfile_multicolumn(zip(X, y, numFrames), os.path.join(constants.PATH_UCFCRIME2LOCAL_README, 'All_data.txt'))
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

def rebuild_load_data():
    normal_videos = read_file(os.path.join(constants.PATH_UCFCRIME2LOCAL_README, 'normal_videos.txt'))
    normal_videos_from_anormal = read_file(os.path.join(constants.PATH_UCFCRIME2LOCAL_README, 'normal_videos_from_anormal.txt'))
    anomaly_normal = read_file(os.path.join(constants.PATH_UCFCRIME2LOCAL_README, 'anomaly_normal.txt'))
    violence_videos = read_file(os.path.join(constants.PATH_UCFCRIME2LOCAL_README, 'violence_videos.txt'))
    X, y, numFrames = [], [], []
    ll = [violence_videos, anomaly_normal]
    llabels = [1, 0]
    for i, videos_list in enumerate(ll):
        for v_video in videos_list:
            if llabels[i] == 1:
                v_path = os.path.join(constants.PATH_UCFCRIME2LOCAL_FRAMES_VIOLENCE, v_video)
            else:
                v_path = os.path.join(constants.PATH_UCFCRIME2LOCAL_FRAMES_NONVIOLENCE, v_video)
            l_frames = os.listdir(v_path)
            X.append(v_video)
            y.append(llabels[i])
            numFrames.append(len(l_frames))

    save_csvfile_multicolumn(zip(X, y, numFrames), os.path.join(constants.PATH_UCFCRIME2LOCAL_README, 'All_data.txt'))
    return X, y, numFrames


    # print('normal_videos', len(normal_videos), normal_videos[0])
    # print('normal_videos_from_anormal', len(normal_videos_from_anormal), normal_videos_from_anormal[0])
    # print('anomaly_normal', len(anomaly_normal), anomaly_normal[0])
    # print('violence_videos', len(violence_videos), violence_videos[0])

def temporal_cut_long_videos():
    anomaly_normal = read_file(os.path.join(constants.PATH_UCFCRIME2LOCAL_README, 'anomaly_normal.txt'))
    for av in anomaly_normal:
        v_path = os.path.join(constants.PATH_UCFCRIME2LOCAL_FRAMES_NONVIOLENCE, av)
        print(v_path)
        l_frames = os.listdir(v_path)
        l_frames.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
        bdx_file_path = os.path.join(constants.PATH_UCFCRIME2LOCAL_Txt_ANNOTATIONS, av+'.txt')
        data = []
        with open(bdx_file_path, 'r') as file:
            for row in file:
                data.append(row.split())
        # data = np.array(data)
        gt_bboxes = []
        anomaly_clips = []
        counter=0
        start=False
        end=False

        clip_frames = []
        counter = 0
        for i,frame_path in enumerate(l_frames):
            # print('------frame_path=',frame_path)
            pth, frame_name = os.path.split(frame_path)
            splits = re.split('(\d+)', frame_name)
            frame_number = int(splits[1])

            if frame_number >= len(data):
                break
            frame_data = data[frame_number]
            # print('video={}, frame={}, frame_number={}, gt={}'.format(video_name, frame_name, frame_number, frame_data))
            if frame_number != int(frame_data[5]):
                print('=========*********** Error en Ground Truth!!!!!!!!!')
                break
            x0, y0, w, h = int(frame_data[1]), int(frame_data[2]), int(frame_data[3])-int(frame_data[1]), int(frame_data[4])-int(frame_data[2])
            gt_bboxes.append([x0, y0, w, h])
            # frame = cv2.imread(os.path.join(v_path,frame_path))
            flac = int(frame_data[6])
            if flac == 0:
                # cv2.rectangle(frame, (x0, y0),(x0+w, y0+h), (0,255,0), 2)
                clip_frames.append(os.path.join(v_path,frame_path))
                if not start:
                    # print('================')
                    # clip_frames.append(os.path.join(v_path,frame_path))
                    start=True
            elif start:
                end=True
                start = False
                # for f in clip_frames:
                #     print(f)
                anomaly_clips.append(clip_frames)
                counter+=1
                f_name = av+'-VSplit-'+str(counter)
                if not os.path.isdir(os.path.join(constants.PATH_UCFCRIME2LOCAL_FRAMES_NONVIOLENCE, f_name)):
                    os.mkdir(os.path.join(constants.PATH_UCFCRIME2LOCAL_FRAMES_NONVIOLENCE, f_name))
                for fimg in clip_frames:
                    newPath = shutil.copy(fimg, os.path.join(constants.PATH_UCFCRIME2LOCAL_FRAMES_NONVIOLENCE, f_name))



                clip_frames=[]
            # cv2.imshow("frame", frame)
            # key = cv2.waitKey(0)

def plot_bbox_annotations():
    anomaly_normal = read_file(os.path.join(constants.PATH_UCFCRIME2LOCAL_README, 'anomaly_normal.txt'))
    for av in anomaly_normal:
        v_path = os.path.join(constants.PATH_UCFCRIME2LOCAL_FRAMES_NONVIOLENCE, av)
        print(v_path)
        l_frames = os.listdir(v_path)
        l_frames.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
        av = av[:-8]
        bdx_file_path = os.path.join(constants.PATH_UCFCRIME2LOCAL_Txt_ANNOTATIONS, av+'.txt')
        data = []
        with open(bdx_file_path, 'r') as file:
            for row in file:
                data.append(row.split())
        # data = np.array(data)
        gt_bboxes = []
        counter=0
        start=False
        end=False

        for i,frame_path in enumerate(l_frames):
            # print('------frame_path=',frame_path)
            pth, frame_name = os.path.split(frame_path)
            splits = re.split('(\d+)', frame_name)
            frame_number = int(splits[1])

            if frame_number >= len(data):
                break
            frame_data = data[frame_number]
            # print('video={}, frame={}, frame_number={}, gt={}'.format(video_name, frame_name, frame_number, frame_data))
            if frame_number != int(frame_data[5]):
                print('=========*********** Error en Ground Truth!!!!!!!!!')
                break
            x0, y0, w, h = int(frame_data[1]), int(frame_data[2]), int(frame_data[3])-int(frame_data[1]), int(frame_data[4])-int(frame_data[2])
            gt_bboxes.append([x0, y0, w, h])
            frame = cv2.imread(os.path.join(v_path,frame_path))
            flac = int(frame_data[6])
            if flac == 0:
                cv2.rectangle(frame, (x0, y0),(x0+w, y0+h), (0,255,0), 2)

            cv2.imshow("frame", frame)
            key = cv2.waitKey(0)

if __name__ == "__main__":
    plot_bbox_annotations()
    # temporal_cut_long_videos()
    # rebuild_load_data()
    # rwf_videos2frames()
    # rwf_load_data()
    # crime2localCreateSplits()
    # getBBoxFile('/Users/davidchoqueluqueroman/Desktop/PAPERS-CODIGOS/violencedetection2/DATASETS/CrimeViolence2LocalDATASET/frames/violence/Arrest003-VSplit1')
