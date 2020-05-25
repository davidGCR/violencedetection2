import numpy as np
import os
import torch
from UTIL.util import save_file, read_file

def partitions(number, k):
    """
    Distribution of the folds
    Args:
        number: number of patients
        k: folds number
    """
    n_partitions = np.ones(k) * int(number / k)
    n_partitions[0 : (number % k)] += 1
    return n_partitions


def get_indices(n_splits, subjects):
    """
    Indices of the set test
    Args:
        n_splits: folds number
        subjects: number of patients
        frames: length of the sequence of each patient
    """
    l = partitions(subjects, n_splits)

    # fold_sizes = l * frames
    # indices = np.arange(subjects * frames).astype(int)
    indices = np.arange(subjects).astype(int)
    current = 0
    for fold_size in l:
        start = current
        stop = current + fold_size
        current = stop
        yield (indices[int(start) : int(stop)])


def k_folds(n_splits, subjects, splits_folder):
    """
    Generates folds for cross validation
    Args:
        n_splits: folds number
        subjects: number of patients
        frames: length of the sequence of each patient
    """
    indices = np.arange(subjects).astype(int)
    if n_splits == 1:
        for i in range(1):
            train, test = train_test_split(80, 20, subjects)
            yield train, test
    # indices = np.arange(subjects * frames).astype(int)
    else:
        if not os.path.exists(os.path.join(splits_folder, 'fold_1_train.txt')):
            for fold,test_idx in enumerate(get_indices(n_splits, subjects)):
                train_idx = np.setdiff1d(indices, test_idx)
                save_file(train_idx, os.path.join(splits_folder, 'fold_' + str(fold + 1) + '_train.txt'))
                save_file(test_idx, os.path.join(splits_folder, 'fold_' + str(fold + 1) + '_test.txt'))
                yield train_idx, test_idx
        else:
            for fold in range(n_splits):
                train_idx = read_file(os.path.join(splits_folder, 'fold_' + str(fold + 1) + '_train.txt'))
                test_idx = read_file(os.path.join(splits_folder, 'fold_' + str(fold + 1) + '_test.txt'))
                yield train_idx, test_idx
            

def train_test_split(s1, s2, len_indices):
    indices = np.arange(len_indices).astype(int)
    train_size = int((s1/100) * len_indices)
    # test_size = dataset_len - train_size
    # train_dataset, test_dataset = torch.utils.data.random_split(fulldataset, [train_size, test_size])
    train_dataset = indices[0:train_size]
    test_dataset = indices[train_size:len_indices]

    return train_dataset, test_dataset


def build_paths(gpath, list_paths):
    for idx,path in enumerate(list_paths):
        list_paths[idx] = gpath + str(list_paths[idx])
    return list_paths

def getAllPaths(gpath, test_idx):
    all_paths = []
    all_labels = []
    llist = os.listdir(gpath)
    for subfolder in llist:
        if subfolder != str(test_idx):
            print(subfolder)
            gpath_violence = gpath + "/" + subfolder + "/Violence/"
            gpath_noviolence = gpath + "/" + subfolder + "/NonViolence/"

            test_paths_violence = os.listdir(gpath_violence)
            test_paths_noviolence = os.listdir(gpath_noviolence)

            test_paths_violence = build_paths(gpath_violence,test_paths_violence)
            test_paths_noviolence = build_paths(gpath_noviolence,test_paths_noviolence)
            all_paths = all_paths + test_paths_violence + test_paths_noviolence
            all_labels = (
                all_labels
                + list([1] * len(test_paths_violence))
                + list([0] * len(test_paths_noviolence))
            )
    
    return all_paths, all_labels

def k_folds_from_folders(gpath, n_splits):
    #     folds = np.arange(1,n_splits+1)
    folds = os.listdir(gpath)
    test_videos = []
    test_labels = []

    for test_folder in folds:
        gpath_violence = gpath + '/' + test_folder + "/Violence/"
        gpath_noviolence = gpath + '/' + test_folder + "/NonViolence/"
        test_paths_violence = os.listdir(gpath_violence)
        test_paths_noviolence = os.listdir(gpath_noviolence)
        
        test_paths_violence = build_paths(gpath_violence, test_paths_violence)
        test_paths_noviolence = build_paths(gpath_noviolence, test_paths_noviolence)
        test_videos = test_paths_violence + test_paths_noviolence
        
        test_labels = list([1] * len(test_paths_violence)) + list(
            [0] * len(test_paths_noviolence)
        )

        train_videos, train_labels = getAllPaths(gpath, test_folder)
        #         train_idx = np.setdiff1d(indices, test_folder)

        yield train_videos, train_labels, test_videos, test_labels


# def __main__():
#     print('test23 kfolfs...')
#     num_folds = 3
#     for train_idx, test_idx in k_folds(n_splits = 3, subjects = 20):
#         print('train_idx',len(train_idx))
#         print(train_idx)
#         print('test_idx',len(test_idx))
#         print(test_idx)
# dataset_train = NNDataset(indices = train_idx)
# dataset_test = NNDataset(indices = test_idx)
# train_loader = torch.utils.data.DataLoader(dataset = dataset_train, batch_size = batch_size_train, **kwargs)
# test_loader = torch.utils.data.DataLoader(dataset = dataset_test, batch_size = batch_size_test, **kwargs)
# for epoch in range(1, num_epochs + 1):
#     train(model, optimizer, epoch, device, train_loader, log_interval)
#     test(model, device, test_loader)


# __main__()
# get_indices()
