
import torch
import os
import glob
from violenceDataset import ViolenceDataset
from violenceDatasetOnline import ViolenceOnlineDataset
from MaskDataset import MaskDataset
# import Saliency.saliencyModel as saliencyModel
import constants
import util
from operator import itemgetter
import kfolds
import random
import numpy as np

def get_test_dataloader(batch_size, num_workers, debugg_mode, numDiPerVideos, dataset_source, data_transforms, interval_duration, avgmaxDuration, shuffle = False):
    """ Get test dataloader to saliency test """
    datasetAll, labelsAll, numFramesAll = createDataset(constants.PATH_HOCKEY_FRAMES_VIOLENCE, constants.PATH_HOCKEY_FRAMES_NON_VIOLENCE) #ordered
    _, _, test_x, test_y = train_test_split_saliency(datasetAll,labelsAll, numFramesAll)
    print(len(test_x), len(test_y), len(numFramesAll))

    image_dataset = ViolenceDataset( dataset=test_x, labels=test_y, spatial_transform=data_transforms["test"], source=dataset_source,
            interval_duration=interval_duration, difference=3, maxDuration=avgmaxDuration, nDynamicImages=numDiPerVideos, debugg_mode=debugg_mode, )
    dataloader = torch.utils.data.DataLoader( image_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, ),
    return test_x, test_y, dataloader

def train_test_split_saliency(datasetAll, labelsAll, numFramesAll, folds_number=1):
    """ Create or load train-test split using kfolds=1 """
    train_x, train_y, test_x, test_y = None, None, None, None
    train_x_path = os.path.join(constants.PATH_SALIENCY_DATASET, 'train_x.txt')
    train_y_path = os.path.join(constants.PATH_SALIENCY_DATASET, 'train_y.txt')
    test_x_path = os.path.join(constants.PATH_SALIENCY_DATASET, 'test_x.txt')
    test_y_path = os.path.join(constants.PATH_SALIENCY_DATASET, 'test_y.txt')
    if not os.path.exists(train_x_path) or not os.path.exists(train_y_path) or not os.path.exists(test_x_path) or not os.path.exists(test_y_path):
        print('Creating Dataset...')
        for train_idx, test_idx in kfolds.k_folds(n_splits=folds_number, subjects=len(datasetAll)):
            combined = list(zip(datasetAll, labelsAll, numFramesAll))
            random.shuffle(combined)
            datasetAll[:], labelsAll[:], numFramesAll[:] = zip(*combined)
            train_x = list(itemgetter(*train_idx)(datasetAll))
            train_y = list(itemgetter(*train_idx)(labelsAll))
            test_x = list(itemgetter(*test_idx)(datasetAll))
            test_y = list(itemgetter(*test_idx)(labelsAll))
            util.saveList2(train_x_path,train_x)
            util.saveList2(train_y_path,train_y)
            util.saveList2(test_x_path,test_x)
            util.saveList2(test_y_path, test_y)
    else:
        print('Loading Dataset...')
        train_x = util.loadList(train_x_path)
        train_y = util.loadList(train_y_path)
        test_x = util.loadList(test_x_path)
        test_y = util.loadList(test_y_path)

    # tx, ty = np.array(train_y), np.array(test_y)
    # unique, counts = np.unique(tx, return_counts=True)
    # print('train_balance: ', dict(zip(unique, counts)))
    # unique, counts = np.unique(ty, return_counts=True)
    # print('test_balance: ', dict(zip(unique, counts)))

    return train_x, train_y, test_x, test_y

def createDatasetViolence(path):  #only violence videos
    """ Create Violence dataset with only violence videos """

    imagesF = []
    list_violence = os.listdir(path)
    list_violence.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    for target in list_violence:
        d = os.path.join(path, target)
        imagesF.append(d)
    labels = list([1] * len(imagesF))
    numFrames = [len(glob.glob1(imagesF[i], "*.jpg")) for i in range(len(imagesF))]
    return imagesF, labels, numFrames

def print_balance(train_y, test_y):
    tx, ty = np.array(train_y), np.array(test_y)
    unique, counts = np.unique(tx, return_counts=True)
    print('train_balance: ', dict(zip(unique, counts)))
    unique, counts = np.unique(ty, return_counts=True)
    print('test_balance: ', dict(zip(unique, counts)))

def createDataset(path_violence, path_noviolence, suffle):
    """ Create Violence dataset with paths and labels """
    imagesF = []

    list_violence = os.listdir(path_violence)
    list_violence.sort(key=lambda f: int("".join(filter(str.isdigit, f))))

    for target in list_violence:
        d = os.path.join(path_violence, target)
        imagesF.append(d)
    imagesNoF = []
    list_no_violence = os.listdir(path_noviolence)
    list_no_violence.sort(key=lambda f: int("".join(filter(str.isdigit, f))))

    for target in list_no_violence:
        d = os.path.join(path_noviolence, target)
        imagesNoF.append(d)

    Dataset = imagesF + imagesNoF
    Labels = list([1] * len(imagesF)) + list([0] * len(imagesNoF))
    NumFrames = [len(glob.glob1(Dataset[i], "*.jpg")) for i in range(len(Dataset))]
    #Shuffle data
    if suffle:
        combined = list(zip(Dataset, Labels, NumFrames))
        random.shuffle(combined)
        Dataset[:], Labels[:], NumFrames[:] = zip(*combined)

    print('Dataset, Labels, NumFrames: ', len(Dataset), len(Labels), len(NumFrames))
    return Dataset, Labels, NumFrames

# def getViolenceDataLoader(x, y, data_transform, numDiPerVideos, dataset_source, avgmaxDuration, interval_duration, batch_size, num_workers, debugg_mode):
#     """ Get Dataloader for violence dataset """
#     dataset = ViolenceDataset( dataset=x, labels=y, spatial_transform=data_transform, source=dataset_source,
#             interval_duration=interval_duration,difference=3, maxDuration=avgmaxDuration, nDynamicImages=numDiPerVideos, debugg_mode=debugg_mode, )
#     dataloader = torch.utils.data.DataLoader( dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
#     return dataloader

def getTrainDataLoader(train_x, train_y, train_numFrames, data_transforms, numDiPerVideos, batch_size, num_workers, videoSegmentLength, positionSegment):
    """ Get train - test dataloaders for violence dataset or masked dataset """
    # image_datasets = None
    # dataset, labels, numFrames, spatial_transform, numDynamicImagesPerVideo, videoSegmentLength, positionSegment
    image_datasets = ViolenceDataset(dataset=train_x, labels=train_y, numFrames=train_numFrames, spatial_transform=data_transforms["train"],
            numDynamicImagesPerVideo=numDiPerVideos,
            videoSegmentLength= videoSegmentLength, positionSegment = positionSegment )
       
    dataloader= torch.utils.data.DataLoader( image_datasets, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        
    return dataloader

def getDataLoaders(train_x, train_y, train_numFrames, test_x, test_y, test_numFrames, data_transforms, numDiPerVideos, batch_size, num_workers, videoSegmentLength, positionSegment):
    """ Get train - test dataloaders for violence dataset or masked dataset """
    image_datasets = None
    image_datasets = {
        #dataset, labels, numFrames, spatial_transform, numDynamicImagesPerVideo, videoSegmentLength, positionSegment
        "train": ViolenceDataset(dataset=train_x, labels=train_y, numFrames=train_numFrames, spatial_transform=data_transforms["train"], numDynamicImagesPerVideo=numDiPerVideos,
         videoSegmentLength= videoSegmentLength, positionSegment = positionSegment ),
        "val": ViolenceDataset(dataset=test_x, labels=test_y, numFrames=test_numFrames , spatial_transform=data_transforms["val"], numDynamicImagesPerVideo=numDiPerVideos,
        videoSegmentLength= videoSegmentLength, positionSegment = positionSegment )
    }
    

    dataloaders_dict = {
        "train": torch.utils.data.DataLoader( image_datasets["train"], batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        "val": torch.utils.data.DataLoader( image_datasets["val"], batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
    }
    return dataloaders_dict

def getOnlineDataLoader(datasetAll, labelsAll, numFramesAll, transform, numDiPerVideos, batch_size, num_workers, overlapping):
    """ Get train - test dataloaders for violence dataset or masked dataset """
    image_datasets = None

    # dataset, labels, numFrames, spatial_transform, nDynamicImages,
    #                 videoSegmentLength, positionSegment, overlapping
    dataset = ViolenceOnlineDataset(dataset=datasetAll, labels=labelsAll, numFrames=numFramesAll, spatial_transform=transform, numDynamicImagesPerVideo=numDiPerVideos,
                                            overlapping =overlapping )
    

    dataloader = torch.utils.data.DataLoader( image_datasets, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dataloader, dataset