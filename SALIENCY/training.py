import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from torch.autograd import Variable
from scipy import misc
from SALIENCY.saliencyModel import build_saliency_model
from tqdm import tqdm
from operator import itemgetter
# import transforms
from loss import Loss
from torch.optim import lr_scheduler
import os
# import util
import argparse
import constants
from constants import DEVICE
# import ANOMALYCRIME.transforms_anomaly as transforms_anomaly

import ANOMALYCRIME.anomalyInitializeDataset as anomaly_initializeDataset
from VIOLENCE_DETECTION.transforms import hockeyTransforms
from VIOLENCE_DETECTION.datasetsMemoryLoader import hockeyLoadData
# from Models import AlexNet

def save_checkpoint(state, filename='sal.pth.tar'):
    print('save in: ',filename)
    torch.save(state, filename)

def load_checkpoint(net,optimizer,filename='small.pth.tar'):
    checkpoint = torch.load(filename)
    net.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return net,optimizer

def train(num_classes, num_epochs, regularizers, device, checkpoint_path, dataloaders_dict, black_box_model, numDynamicImages):
    saliency_m = build_saliency_model(num_classes)
    saliency_m = saliency_m.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(saliency_m.parameters())           
    classifier = torch.load(black_box_model)
    # classifier.inferenceMode()

    loss_func = Loss(num_classes=num_classes, regularizers=regularizers)
    best_loss = 1000.0
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        print("----- Epoch {}/{}".format(epoch+1, num_epochs))
        running_loss = 0.0
        running_loss_train = 0.0
        for i, data in tqdm(enumerate(dataloaders_dict['train'], 0)):
            # get the inputs
            inputs, labels, video_name, bbox_segments = data #dataset load [bs,ndi,c,w,h]
            # print('dataset element: ',inputs_r.shape) #torch.Size([8, 1, 3, 224, 224])
            # if numDynamicImages > 1:
            #     inputs = inputs.permute(1, 0, 2, 3, 4)
            #     inputs = torch.squeeze(inputs, 0)
            # wrap them in Variable
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            # zero the parameter gradients
            optimizer.zero_grad()

            mask, out = saliency_m(inputs, labels)
           
            loss = loss_func.get(mask,inputs,labels,black_box_model)
            # running_loss += loss.data[0]
            running_loss += loss.item()
            running_loss_train += loss.item()*inputs.size(0)
            loss.backward()
            optimizer.step()

        epoch_loss = running_loss / len(dataloaders_dict["train"].dataset)
        epoch_loss_train = running_loss_train / len(dataloaders_dict["train"].dataset)
        print("{} RawLoss: {:.4f} Loss: {:.4f}".format('train', epoch_loss, epoch_loss_train))

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            print('Saving entire saliency model...')
            save_checkpoint(saliency_m,checkpoint_path)

def __anomaly_main__():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batchSize", type=int, default=8)
    parser.add_argument("--numEpochs", type=int, default=10)
    parser.add_argument("--numWorkers", type=int, default=4)
    parser.add_argument("--areaL", type=float, default=8)
    parser.add_argument("--smoothL", type=float, default=0.5)
    parser.add_argument("--preserverL", type=float, default=0.3)
    parser.add_argument("--areaPowerL", type=float, default=0.3)
    parser.add_argument("--blackBoxFile", type=str)  #areaL-9.0_smoothL-0.3_epochs-20
    parser.add_argument("--videoSegmentLength", type=int, default=0)
    parser.add_argument("--positionSegment", type=str, default='random')
    parser.add_argument("--maxNumFramesOnVideo", type=int, default=0)
    parser.add_argument("--shuffle", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--numDiPerVideos", type=int)
    
    args = parser.parse_args()
    
    input_size = 224
    transforms = hockeyTransforms(input_size)
    num_classes = 2
    
    regularizers = {'area_loss_coef': args.areaL,
                    'smoothness_loss_coef': args.smoothL,
                    'preserver_loss_coef': args.preserverL,
                    'area_loss_power': args.areaPowerL}
    datasetAll, labelsAll, numFramesAll = hockeyLoadData()
    train_x, train_y, train_numFrames, test_x, test_y, test_numFrames = hockeyTrainTestSplit('train-test-1', datasetAll, labelsAll, numFramesAll)
    default_args = {
            'X': train_x,
            'y': train_y,
            'numFrames': train_numFrames,
            'transform': transforms['train'],
            'NDI': 1,
            'videoSegmentLength': 30,
            'positionSegment': 'begin',
            'overlapping': 0,
            'frameSkip': 0,
            'skipInitialFrames': 0,
            'batchSize': 8,
            'shuffle': True,
            'numWorkers': 4,
            'pptype': None,
            'modelType': 'resnet50'
    }
    train_dt_loader = MyDataloader(default_args)
    # path_dataset = constants.PATH_UCFCRIME2LOCAL_FRAMES_REDUCED
    # train_videos_path = os.path.join(constants.PATH_UCFCRIME2LOCAL_README, 'Train_split_AD.txt')
    # test_videos_path = os.path.join(constants.PATH_UCFCRIME2LOCAL_README, 'Test_split_AD.txt')
    # dataloaders_dict, test_names = anomaly_initializeDataset.initialize_final_anomaly_dataset(path_dataset, train_videos_path, test_videos_path,
    #                                 batch_size, num_workers, numDiPerVideos, transforms_dataset, maxNumFramesOnVideo, videoSegmentLength, positionSegment, shuffle,0)
    
    # checkpoint_path = os.path.join(saliency_model_folder,'mask_model_'+str(videoSegmentLength)+'_frames_di_'+checkpoint_info + '_epochs-' + str(num_epochs) + '.tar')
    
    train(num_classes, num_epochs, regularizers, device, checkpoint_path, dataloaders_dict, black_box_file, numDiPerVideos)

__anomaly_main__()