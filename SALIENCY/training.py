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

# import ANOMALYCRIME.anomalyInitializeDataset as anomaly_initializeDataset
from VIOLENCE_DETECTION.transforms import hockeyTransforms
from VIOLENCE_DETECTION.datasetsMemoryLoader import hockeyLoadData, hockeyTrainTestSplit
from UTIL.chooseModel import initialize_model
from VIOLENCE_DETECTION.dataloader import MyDataloader
# from Models import AlexNet

def save_checkpoint(state, filename='sal.pth.tar'):
    print('save in: ',filename)
    torch.save(state, filename)

def load_checkpoint(net,optimizer,filename='small.pth.tar'):
    checkpoint = torch.load(filename)
    net.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return net,optimizer

def train(mask_model, criterion, optimizer, regularizers, classifier_model, num_epochs, dataloader, numDynamicImages, checkpoint_path):
    loss_func = Loss(num_classes=2, regularizers=regularizers, num_dynamic_images=numDynamicImages)
    best_loss = 1000.0
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        print("----- Epoch {}/{}".format(epoch+1, num_epochs))
        running_loss = 0.0
        running_loss_train = 0.0
        for data in tqdm(dataloader):
            # get the inputs
            # dynamicImages, label, vid_name, preprocessing_time
            inputs, labels, video_name, _ = data  #dataset load [bs,ndi,c,w,h]
            # print('Inputs=', inputs.size())
            # print('dataset element: ',inputs_r.shape) #torch.Size([8, 1, 3, 224, 224])
            # if numDynamicImages > 1:
            #     inputs = inputs.permute(1, 0, 2, 3, 4)
            #     inputs = torch.squeeze(inputs, 0)
            # wrap them in Variable
            inputs, labels = Variable(inputs.to(DEVICE)), Variable(labels.to(DEVICE))
            # zero the parameter gradients
            optimizer.zero_grad()
            mask, out = mask_model(inputs, labels)
            print('MAsk passed=', mask.size())
            loss = loss_func.get(mask, inputs, labels, classifier_model)
            # running_loss += loss.data[0]
            running_loss += loss.item()
            running_loss_train += loss.item()*inputs.size(0)
            loss.backward()
            optimizer.step()

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_loss_train = running_loss_train / len(dataloader.dataset)
        print("{} RawLoss: {:.4f} Loss: {:.4f}".format('train', epoch_loss, epoch_loss_train))

        if checkpoint_path is not None and epoch_loss < best_loss:
            best_loss = epoch_loss
            print('Saving model...',checkpoint_path+'-epoch-'+str(epoch)+'.pth')
            torch.save(mask_model.state_dict(), checkpoint_path+'-epoch-'+str(epoch)+'-loss='+str(epoch_loss)+'.pth')
        #     print('Saving entire saliency model...')
        #     save_checkpoint(saliency_m,checkpoint_path)

def __anomaly_main__():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier", type=str)
    parser.add_argument("--modelType", type=str)
    parser.add_argument("--batchSize", type=int, default=8)
    parser.add_argument("--numEpochs", type=int, default=10)
    parser.add_argument("--numWorkers", type=int, default=4)
    parser.add_argument("--areaL", type=float, default=8)
    parser.add_argument("--smoothL", type=float, default=0.5)
    parser.add_argument("--preserverL", type=float, default=0.3)
    parser.add_argument("--areaPowerL", type=float, default=0.3)
    parser.add_argument("--numDiPerVideos", type=int)
    parser.add_argument("--saveCheckpoint",type=lambda x: (str(x).lower() == 'true'), default=False)
    
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
            'NDI': args.numDiPerVideos,
            'videoSegmentLength': 20,
            'positionSegment': 'begin',
            'overlapping': 0,
            'frameSkip': 0,
            'skipInitialFrames': 0,
            'batchSize': args.batchSize,
            'shuffle': True,
            'numWorkers': args.numWorkers,
            'pptype': None,
            'modelType': args.modelType
    }
    train_dt_loader = MyDataloader(default_args)
    mask_model = build_saliency_model(num_classes)
    mask_model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mask_model.parameters())           
    # classifier = torch.load(black_box_model)
    classifier_model, _ = initialize_model(model_name=args.modelType,
                                            num_classes=2,
                                            feature_extract=False,
                                            numDiPerVideos=args.numDiPerVideos,
                                            joinType='maxTempPool',
                                            use_pretrained=True)
    if DEVICE == 'cuda:0':
        classifier_model.load_state_dict(torch.load(args.classifier), strict=False)
    else:
        classifier_model.load_state_dict(torch.load(args.classifier, map_location=DEVICE))
    classifier_model.to(DEVICE)
    classifier_model.eval()
    checkpoint_path = None
    if args.saveCheckpoint:
        checkpoint_path = 'MaskModel_backnone={}_NDI={}_AreaLoss={}_SmoothLoss={}_PreservLoss={}_AreaLoss2={}_epochs={}'.format(args.modelType,
                                                                                                                                args.numDiPerVideos,
                                                                                                                                args.areaL,
                                                                                                                                args.smoothL,
                                                                                                                                args.preserverL,
                                                                                                                                args.areaPowerL,
                                                                                                                                args.numEpochs)
        checkpoint_path = os.path.join(constants.PATH_RESULTS,'MASKING','checkpoints',checkpoint_path)                                                                                                                                
    train(mask_model=mask_model,
          criterion=criterion,
          optimizer=optimizer,
          regularizers=regularizers,
          classifier_model=classifier_model,
          num_epochs=args.numEpochs,
          dataloader=train_dt_loader.dataloader,
          numDynamicImages=args.numDiPerVideos,
          checkpoint_path=checkpoint_path)
    # train(num_classes, args.num_epochs, regularizers, train_dt_loader, args.classifier, args.numDiPerVideos)

__anomaly_main__()