import sys
sys.path.insert(1, "/media/david/datos/PAPERS-SOURCE_CODE/MyCode")
import torch 
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 
import numpy as np 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
import torch.optim as lr_scheduler
import os 
import glob 
import cv2 
from PIL import Image
from torch.utils.data import Dataset
from torchvision import models 
import torch 
from torch.autograd import Variable 
import time
import numpy as np

from AlexNet import *
from ViolenceDatasetV2 import *
from trainer import *
from kfolds import *
from operator import itemgetter
import random
from initializeModel import *
from util import *
from verifyParameters import *

from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.model_selection import cross_val_score
from transforms import *
import argparse


def init(
    path_violence,
    path_noviolence,
    path_results,
    modelType,
    ndi,
    num_workers,
    data_transforms,
    dataset_source,
    interval_duration,
    avgmaxDuration,
    batch_size,
    num_epochs,
    feature_extract,
    joinType,
    scheduler_type,
    device,
    criterion,
    debugg_mode=False):

    datasetAll, labelsAll, numFramesAll = createDataset(path_violence, path_noviolence)
    combined = list(zip(datasetAll, labelsAll, numFramesAll))
    random.shuffle(combined)
    datasetAll[:], labelsAll[:], numFramesAll[:] = zip(*combined)
    print(len(datasetAll), len(labelsAll), len(numFramesAll))

    dataset_train = datasetAll
    dataset_train_labels =  labelsAll 

    # dataset_test = list(itemgetter(*te_idx)(datasetAll)) 
    # dataset_test_labels =  list(itemgetter(*te_idx)(labelsAll))

    image_datasets = {
        'train':ViolenceDatasetVideos(
            dataset= dataset_train,
            labels=dataset_train_labels,
            spatial_transform = data_transforms['train'],
            source = dataset_source,
            interval_duration = interval_duration,
            difference = 3,
            maxDuration = avgmaxDuration,
            nDynamicImages = ndi,
            debugg_mode = debugg_mode
        )
    }
    dataloaders_dict = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=False, num_workers=num_workers),
    #     'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=True, num_workers=num_workers)
    }

    model = None
    model, input_size = initialize_model(model_name = modelType, num_classes = 2, feature_extract=True, numDiPerVideos=ndi, joinType = joinType ,use_pretrained=True)
    model.eval()
    model.to(device)
    params_to_update = verifiParametersToTrain(model)
    # Observe that all parameters are being optimized
    optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

    lista = []
    dataset_labels = None

    for inputs, labels in dataloaders_dict["train"]:
        print('==== dataloader size:' ,inputs.size())
        inputs = inputs.permute(1, 0, 2, 3, 4)
        inputs = inputs.to(device)
        dataset_labels = labels
        # zero the parameter gradient
        optimizer.zero_grad()
        with torch.set_grad_enabled(False):
            if joinType == 'cat':
                outputs = model.getFeatureVectorCat(inputs)
            elif joinType == 'tempMaxPool':
                outputs = model.getFeatureVectorTempPool(inputs)
        # outputs = model(inputs)
        outputs = outputs.cpu()
        lista.append(outputs.numpy())

    dataset = np.array(lista)
    dataset = dataset.squeeze()

    # path_results = '/media/david/datos/Violence DATA/HockeyFights/CNN+SVM data'
    saveData(
        path_results,
        modelType,
        "dataset",
        ndi,
        dataset_source,
        feature_extract,
        joinType,
        dataset,
    )
    saveData(
        path_results,
        modelType,
        "dataset_labels",
        ndi,
        dataset_source,
        feature_extract,
        joinType,
        dataset_labels,
        )


# print(dataset)
# print(dataset_paths[0:10])
# print(labels)

# # Grid Search
# # Parameter Grid
# param_grid = {'C': [0.1, 1, 10, 100],'gamma': [1, 0.1, 0.01, 0.001, 0.00001, 10]}
# # Make grid search classifier
# clf_grid = GridSearchCV(svm.SVC(), param_grid, verbose=1)
# # Train the classifier
# clf_grid.fit(dataset, labels)
# # clf = grid.best_estimator_()
# print("Best Parameters:", clf_grid.best_params_)
# print("Best Estimators:", clf_grid.best_estimator_)



def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_violence",type=str,default="/media/david/datos/Violence DATA/HockeyFights/frames/violence",help="Directory containing violence videos")
    parser.add_argument("--path_noviolence",type=str,default="/media/david/datos/Violence DATA/HockeyFights/frames/nonviolence",help="Directory containing non violence videos")
    parser.add_argument("--path_results",type=str,default="/media/david/datos/Violence DATA/HockeyFights/CNN+SVM data",help="Directory containing results")
    parser.add_argument("--modelType",type=str,default="alexnet",help="model")
    parser.add_argument("--num_epochs",type=int,default=30)
    parser.add_argument("--batch_size",type=int,default=64)
    parser.add_argument("--feature_extract",type=bool,default=True,help="to fine tunning")
    parser.add_argument("--scheduler_type",type=str,default="",help="learning rate scheduler")
    parser.add_argument("--debugg_mode", type=bool, default=False, help="show prints")
    parser.add_argument("--ndi", type=int, help="num dyn imgs")
    parser.add_argument("--joinType",type=str,default="tempMaxPool",help="show prints")

    args = parser.parse_args()

    # path_models = "/media/david/datos/Violence DATA/HockeyFights/Models"
    path_results = args.path_results
    path_violence = args.path_violence
    path_noviolence = args.path_noviolence
    modelType = args.modelType
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    feature_extract = args.feature_extract
    joinType = args.joinType
    scheduler_type = args.scheduler_type
    debugg_mode = args.debugg_mode
    ndi = args.ndi
    
    dataset_source = "frames"
    debugg_mode = False
    avgmaxDuration = 1.66
    interval_duration = 0.3
    num_workers = 4
    input_size = 224

    transforms_data = createTransforms(input_size)
    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    init(path_violence, path_noviolence, path_results, modelType, ndi, num_workers, transforms_data,
            dataset_source, interval_duration, avgmaxDuration, batch_size, num_epochs, feature_extract, joinType, scheduler_type, device, criterion, debugg_mode)
            
    dataset = loadArray(path_results,
        modelType,
        "dataset",
        ndi,
        dataset_source,
        feature_extract,
        joinType
        )
    dataset_labels = loadArray(path_results,
        modelType,
        "dataset_labels",
        ndi,
        dataset_source,
        feature_extract,
        joinType
        )
    labels = dataset_labels.numpy()

    # print('dataset_paths shape:', type(dataset_paths), len(dataset_paths))
    print('dataset shape:', type(dataset), dataset.shape)
    print('labels shape:', type(dataset_labels), dataset_labels.shape)

    clf = svm.SVC(kernel='linear', C=1)
    scores = cross_val_score(clf, dataset, labels, cv=5)
    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

__main__()

# clf2 = svm.SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,
#     decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
#     kernel='rbf', max_iter=-1, probability=False, random_state=None,
#     shrinking=True, tol=0.001, verbose=False)
# scores2 = cross_val_score(clf2, dataset, labels, cv=5)
# print(scores2)

# clf2 = svm.SVC(C=100, cache_size=200, class_weight=None, coef0=0.0,
#     decision_function_shape='ovr', degree=3, gamma=1e-05, kernel='rbf',
#     max_iter=-1, probability=False, random_state=None, shrinking=True,
#     tol=0.001, verbose=False)
# scores2 = cross_val_score(clf2, dataset, labels, cv=5)
# print(scores2)
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores2.mean(), scores2.std() * 2))