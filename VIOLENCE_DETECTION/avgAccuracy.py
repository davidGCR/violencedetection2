import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import DEVICE
from VIOLENCE_DETECTION.rgbDataset import RGBDataset
from MODELS.ViolenceModels import ResNetRGB
from UTIL.kfolds import k_folds
from VIOLENCE_DETECTION.datasetsMemoryLoader import hockeyLoadData
from VIOLENCE_DETECTION.transforms import hockeyTransforms
from VIOLENCE_DETECTION.violenceDataset import ViolenceDataset
from UTIL.chooseModel import initialize_model
import constants
import torch
from operator import itemgetter
import pandas as pd
from tqdm import tqdm
import numpy as np

def load_model(path, stream_type):
    checkpoint = torch.load(path, map_location=DEVICE)
    if stream_type == 'rgb':
        model = checkpoint['model_config']['model']
        featureExtract = checkpoint['model_config']['featureExtract']
        model_ = ResNetRGB(num_classes=2, model_name=model, feature_extract=featureExtract)
        model_ = model_.build_model()
    else:
        model = checkpoint['model_config']['modelType']
        numDynamicImages = checkpoint['model_config']['numDynamicImages']
        joinType = checkpoint['model_config']['joinType']
        featureExtract = checkpoint['model_config']['featureExtract']
        model_, _ = initialize_model(model_name=model,
                                        num_classes=2,
                                        feature_extract=featureExtract,
                                        numDiPerVideos=numDynamicImages,
                                        joinType=joinType,
                                        use_pretrained=True)

    model_.to(DEVICE)
    # print(model_)
    if DEVICE == 'cuda:0':
        model_.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model_.load_state_dict(checkpoint['model_state_dict'])
    return model_

def test_rgb_model(model, dataloader):
    model.eval()
    names = []
    labels_gt = []
    predictions = []
    scores = []
    for data in tqdm(dataloader):
        v_name, inputs, labels = data
        # print('vname=', v_name, '-Label=', labels)
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        # zero the parameter gradients
        # optimizer.zero_grad()
        batch_size = inputs.size()[0]
        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model(inputs) #tensor([[-11.7874,  11.6377]]) torch.Size([1, 2])
            _, preds = torch.max(outputs, 1)
            # o = outputs.cpu().numpy()
            # print(outputs, o, o.shape)
            predictions.append(preds.item())
            labels_gt.append(labels.item())
            names.append(v_name[0])
            scores.append(outputs.cpu().numpy())
    return names, labels_gt, predictions, scores

def test_dyn_model(model, dataloader):
    model.eval()
    names = []
    labels_gt = []
    predictions = []
    scores = []
    for data in tqdm(dataloader):
        # dynamicImages, label, vid_name, preprocessing_time, paths
        inputs, labels, v_name, _ ,  _ = data
        # print('inputs=', inputs.size(), type(inputs))
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        batch_size = inputs.size()[0]
        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            predictions.append(preds.item())
            labels_gt.append(labels.item())
            names.append(v_name[0])
            scores.append(outputs.cpu().numpy())
    return names, labels_gt, predictions, scores

def avg_accuracy(y, y_pred_rgb, y_pred_dyn):
    y_rgb = np.concatenate(y_pred_rgb, axis=0)
    y_dyn = np.concatenate(y_pred_dyn, axis=0)
    preds_sum = y_rgb + y_dyn
    avg_preds = []
    for i in range(preds_sum.shape[0]):
        pred = np.argmax(preds_sum[i])
        avg_preds.append(pred)
    avg_preds = np.array(avg_preds)
    labels = np.array(y)
    corrects = np.sum(y == avg_preds)
    acc = corrects / preds_sum.shape[0]
    print('Accuracy=', acc)
    return acc
    # for fold in range(5):
    #     rgb = 'RGB_Preds_fold-{}.csv'.format(fold+1)
    #     dyn = 'DYN_Preds_fold-{}.csv'.format(fold + 1)
    #     df_rgb = pd.read_csv(rgb)
    #     df_dyn = pd.read_csv(dyn)
    #     # for i in range(20):
    #     #     s_r = df_rgb['Scores'][i]
    #     #     s_d = df_dyn['Scores'][i]
            
    #     #     print(type(s_r), type(s_d))
    #     s_r = df_rgb.to_numpy()
    #     s_d = df_dyn.to_numpy()
    #     print(s_r[11][4], s_d[11][4], type(s_r[11][4]), type(s_d[11][4])) 

def main():
    datasetAll, labelsAll, numFramesAll = hockeyLoadData()
    rgb_transforms = hockeyTransforms(input_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dyn_transforms = hockeyTransforms(input_size=224)
    fold = 0
    folds_number = 5
    shuffle = False
    accs = []
    for train_idx, test_idx in k_folds(n_splits=folds_number, subjects=len(datasetAll), splits_folder=constants.PATH_HOCKEY_README):
        fold = fold + 1
        print("**************** Fold:{}/{} ".format(fold, folds_number))
        train_x, train_y, test_x, test_y = None, None, None, None
        
        test_x = list(itemgetter(*test_idx)(datasetAll))
        test_y = list(itemgetter(*test_idx)(labelsAll))
        test_numFrames = list(itemgetter(*test_idx)(numFramesAll))

        #### RGB ####
        rgb_dataset = RGBDataset(dataset=test_x,
                                labels=test_y,
                                numFrames=test_numFrames,
                                spatial_transform=rgb_transforms['test'],
                                frame_idx=14)
        rgb_dataloader = torch.utils.data.DataLoader(rgb_dataset, batch_size=1, shuffle=shuffle, num_workers=4)
        rgb_path = os.path.join(constants.PATH_RESULTS, 'HOCKEY', 'checkpoints', 'RGBCNN-dataset=\'hockey\'_model=\'resnet50\'_frameIdx=14_numEpochs=25_featureExtract=False_fold=' + str(fold) + '.pt')
        print(rgb_path)
        model_rgb = load_model(rgb_path, 'rgb')
        names, labels_gt, predictions, scores_rgb = test_rgb_model(model_rgb, rgb_dataloader)
        df = pd.DataFrame(list(zip(names, labels_gt, predictions, scores_rgb)), columns=['Name', 'Label', 'RGB_Pred', 'Scores'])
        # df.to_csv('RGB_Preds_fold-{}.csv'.format(fold))
        # print(df.head(20))

        #### Dyn ####
        dyn_path = os.path.join(constants.PATH_RESULTS, 'HOCKEY', 'checkpoints', 'HOCKEY-Model-resnet50,segmentLen-30,numDynIms-1,frameSkip-0,segmentPreprocessing-False,epochs-25,split_type-cross-val,fold-' + str(fold) + '.pt')
        model_dyn = load_model(dyn_path, 'dyn')
        print(dyn_path)
        # print(model_dyn)
        dyn_dataset = ViolenceDataset(dataset=test_x,
                                    labels=test_y,
                                    numFrames=test_numFrames,
                                    spatial_transform=dyn_transforms['test'],
                                    numDynamicImagesPerVideo=1,
                                    videoSegmentLength=30,
                                    positionSegment='begin',
                                    overlaping=0,
                                    frame_skip=0,
                                    skipInitialFrames=0,
                                    ppType=None)
        dyn_dataloader = torch.utils.data.DataLoader(dyn_dataset, batch_size=1, shuffle=shuffle, num_workers=4)
        names, labels_gt, predictions, scores_dyn = test_dyn_model(model_dyn, dyn_dataloader)
        df = pd.DataFrame(list(zip(names, labels_gt, predictions, scores_dyn)), columns=['Name', 'Label', 'DYN_Pred', 'Scores'])
        # df.to_csv('DYN_Preds_fold-{}.csv'.format(fold))
        # print(df.head(20))
        accs.append(avg_accuracy(labels_gt, scores_rgb, scores_dyn))
        # print("Accuracy: %0.3f (+/- %0.3f)," % (np.array(accs).mean(), np.array(accs).std() * 2))
    # print('Test AVG Accuracy={}, Test AVG Loss={}'.format(np.average(cv_test_accs), np.average(cv_test_losses)))
    print("Accuracy: %0.3f (+/- %0.3f)," % (np.array(accs).mean(), np.array(accs).std() * 2))

if __name__ == "__main__":
    main()
    # avg_accuracy()