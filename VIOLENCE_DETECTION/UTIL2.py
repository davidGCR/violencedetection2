
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transforms import hockeyTransforms, vifTransforms, ucf2CrimeTransforms, rwf_2000_Transforms
from datasetsMemoryLoader import hockeyLoadData, hockeyTrainTestSplit, vifLoadData, crime2localLoadData, customize_kfold, rwf_load_data
# from MODELS.ViolenceModels import
from UTIL.chooseModel import initialize_model
import torch
from constants import DEVICE
from MODELS.ViolenceModels import ResNetRGB
import constants
import matplotlib.pyplot as plt
import cv2
import os

def plot_example(X, y, v_names):
    """Plot the first 5 images and their labels in a row."""
    for i in range(5):
        (X_t, idx, imgs, bb) = X[i]
        y_ = y[i]
        # print('**** ', X_t.size(), imgs[0].shape, type(y_), type(vid_name))

        vid_name = v_names[idx]
        _, vid_name = os.path.split(vid_name)
        title = vid_name + '-' + str(y_)
        cv2.imshow(str(title), imgs[0])
        key = cv2.waitKey(0)
        # plt.subplot(151 + i)
        # plt.imshow(imgs[0])
        # plt.xticks([])
        # plt.yticks([])
        # plt.title(y_)

    # X = torch.squeeze(X)
    # X = X.permute(1,2,0)
    # X=X.numpy()
    # print('----X=', X.size())
    # print('----X=', X.shape)
    # cv2.imshow("dyn_img", X)
    # key = cv2.waitKey(0)

def base_dataset(dataset, input_size=224, mean=None, std=None):
    if dataset == 'ucfcrime2local':
        datasetAll, labelsAll, numFramesAll = crime2localLoadData(min_frames=40)
        transforms = ucf2CrimeTransforms(input_size=input_size, mean=mean, std=std)
    elif dataset == 'vif':
        # print('hereeeeeeeeee')
        datasetAll, labelsAll, numFramesAll, splitsLen = vifLoadData(constants.PATH_VIF_FRAMES)
        transforms = vifTransforms(input_size=input_size, mean=mean, std=std)
    elif dataset == 'hockey':
        datasetAll, labelsAll, numFramesAll = hockeyLoadData(shuffle=True)
        transforms = hockeyTransforms(input_size=input_size, mean=mean, std=std)
    elif dataset == 'rwf-2000':
        train_names, train_labels, train_num_frames, test_names, test_labels, test_num_frames = rwf_load_data()
        transforms = rwf_2000_Transforms(input_size=input_size, mean=mean, std=std)
        return train_names, train_labels, train_num_frames, test_names, test_labels, test_num_frames, transforms

    return datasetAll, labelsAll, numFramesAll, transforms

def transforms_dataset(dataset, mean=None, std=None):
    if dataset == 'ucfcrime2local':
        transforms = ucf2CrimeTransforms(224, mean=mean, std=std)
    elif dataset == 'vif':
        transforms = vifTransforms(input_size=224, mean=mean, std=std)
    elif dataset == 'hockey':
        transforms = hockeyTransforms(input_size=224, mean=mean, std=std)
    elif dataset == 'rwf-2000':
        transforms = rwf_2000_Transforms(input_size=224, mean=mean, std=std)

    return transforms


def load_model(modelPath, stream_type):
    checkpoint = torch.load(modelPath, map_location=DEVICE)
    args = dict()
    if stream_type == 'rgb':
        model = checkpoint['model_config']['model']
        # dataset = checkpoint['model_config']['dataset']
        args['featureExtract'] = checkpoint['model_config']['featureExtract']
        args['frameIdx'] = checkpoint['model_config']['frameIdx']
        model_ = ResNetRGB(num_classes=2, model_name=model, feature_extract=args['featureExtract'])
        model_ = model_.build_model()
    else:
        # checkpoint = torch.load(modelPath, map_location=DEVICE)
        model = checkpoint['model_config']['model']


        args['numDynamicImages'] = checkpoint['model_config']['numDynamicImages']
        args['joinType'] = checkpoint['model_config']['joinType']
        args['freezeConvLayers'] = checkpoint['model_config']['freezeConvLayers']
        args['videoSegmentLength'] = checkpoint['model_config']['segmentLength']
        args['overlapping'] = checkpoint['model_config']['overlap']
        args['frameSkip'] = checkpoint['model_config']['frameSkip']
        args['skipInitialFrames'] = checkpoint['model_config']['skipInitialFrames']
        args['useKeyframes'] = checkpoint['model_config']['useKeyframes']
        args['windowLen'] = checkpoint['model_config']['windowLen']

        model_, _ = initialize_model(model_name=model,
                                        num_classes=2,
                                        freezeConvLayers=args['freezeConvLayers'],
                                        numDiPerVideos=args['numDynamicImages'],
                                        joinType=args['joinType'],
                                        use_pretrained=True)

    model_.to(DEVICE)
    # print(model_)
    if DEVICE == 'cuda:0':
        model_.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model_.load_state_dict(checkpoint['model_state_dict'])

    return model_, args
