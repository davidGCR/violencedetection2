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
from VIOLENCE_DETECTION.transforms import hockeyTransforms, ucf2CrimeTransforms
from VIOLENCE_DETECTION.datasetsMemoryLoader import hockeyLoadData, hockeyTrainTestSplit, crime2localLoadData, get_Fold_Data
from UTIL.chooseModel import initialize_model
from UTIL.util import load_torch_checkpoint
from VIOLENCE_DETECTION.dataloader import MyDataloader
# from Models import AlexNet

# def save_checkpoint(state, filename='sal.pth.tar'):
#     print('save in: ',filename)
#     torch.save(state, filename)

# def load_checkpoint(net,optimizer,filename='small.pth.tar'):
#     checkpoint = torch.load(filename)
#     net.load_state_dict(checkpoint['state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer'])
#     return net,optimizer

def train(mask_model, criterion, optimizer, regularizers, classifier_model, num_epochs, dataloader, numDynamicImages, checkpoint_path):
    loss_func = Loss(num_classes=2, regularizers=regularizers, num_dynamic_images=numDynamicImages)
    best_loss = 1000.0
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        print("----- Epoch {}/{}".format(epoch+1, num_epochs))
        running_loss = 0.0
        running_loss_train = 0.0
        for data in tqdm(dataloader):
            inputs, labels, video_name, _, _ = data  #dataset load [bs,ndi,c,w,h]
            # print('Inputs=', inputs.size())
            # wrap them in Variable
            inputs, labels = Variable(inputs.to(DEVICE)), Variable(labels.to(DEVICE))
            # zero the parameter gradients
            optimizer.zero_grad()
            mask, out = mask_model(inputs, labels)
            # print('MAsk passed=', mask.size())
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
            name = '{}_epoch={}.pth'.format(checkpoint_path, epoch)
            print('Saving model...', name)
            torch.save({
                'epoch': epoch,
                'loss': epoch_loss,
                'model_state_dict': mask_model.state_dict(),
                }, checkpoint_path)
            # torch.save(mask_model.state_dict(), name)
        #     print('Saving entire saliency model...')
        #     save_checkpoint(saliency_m,checkpoint_path)

def base_dataset(dataset, fold):
    if dataset == 'UCFCRIME2LOCAL':
        mytransfroms = ucf2CrimeTransforms(224)
        X, y, numFrames = crime2localLoadData(min_frames=40)
        train_idx, test_idx = get_Fold_Data(fold)
        train_x = list(itemgetter(*train_idx)(X))
        train_y = list(itemgetter(*train_idx)(y))
        train_numFrames = list(itemgetter(*train_idx)(numFrames))
        test_x = list(itemgetter(*test_idx)(X))
        test_y = list(itemgetter(*test_idx)(y))
        test_numFrames = list(itemgetter(*test_idx)(numFrames))
    elif dataset == 'HOCKEY':
        mytransfroms = hockeyTransforms(224)
        datasetAll, labelsAll, numFramesAll = hockeyLoadData()
        train_x, train_y, train_numFrames, test_x, test_y, test_numFrames = hockeyTrainTestSplit('train-test-' + str(fold), datasetAll, labelsAll, numFramesAll)
    return train_x, train_y, train_numFrames, mytransfroms

def __anomaly_main__():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier", type=str)
    # parser.add_argument("--modelType", type=str)
    parser.add_argument("--batchSize", type=int, default=8)
    parser.add_argument("--numEpochs", type=int, default=10)
    parser.add_argument("--numWorkers", type=int, default=4)
    parser.add_argument("--areaL", type=float, default=8)
    parser.add_argument("--smoothL", type=float, default=0.5)
    parser.add_argument("--preserverL", type=float, default=0.3)
    parser.add_argument("--areaPowerL", type=float, default=0.3)
    # parser.add_argument("--numDiPerVideos", type=int)
    # parser.add_argument("--segmentLen", type=int)
    parser.add_argument("--saveCheckpoint",type=lambda x: (str(x).lower() == 'true'), default=False)
    
    args = parser.parse_args()
    num_classes = 2

    class_checkpoint = load_torch_checkpoint(args.classifier)
    train_x, train_y, train_numFrames, mytransfroms = base_dataset(class_checkpoint['model_config']['dataset'], fold=1)
    print(class_checkpoint['model_config'])
    
    regularizers = {'area_loss_coef': args.areaL,
                    'smoothness_loss_coef': args.smoothL,
                    'preserver_loss_coef': args.preserverL,
                    'area_loss_power': args.areaPowerL}
    default_args = {
            'X': train_x,
            'y': train_y,
            'numFrames': train_numFrames,
            'transform': mytransfroms['train'],
            'NDI': class_checkpoint['model_config']['numDynamicImages'],
            'videoSegmentLength': class_checkpoint['model_config']['segmentLength'],
            'positionSegment': 'begin',
            'overlapping': class_checkpoint['model_config']['overlap'],
            'frameSkip': class_checkpoint['model_config']['frameSkip'],
            'skipInitialFrames': class_checkpoint['model_config']['skipInitialFrames'],
            'batchSize': args.batchSize,
            'shuffle': True,
            'numWorkers': args.numWorkers,
            'pptype': None,
            'modelType': class_checkpoint['model_config']['modelType']
    }
    train_dt_loader = MyDataloader(default_args)
    mask_model = build_saliency_model(num_classes)
    mask_model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mask_model.parameters())           
    # classifier = torch.load(black_box_model)
    classifier_model, _ = initialize_model(model_name=class_checkpoint['model_config']['modelType'],
                                            num_classes=2,
                                            feature_extract=False,
                                            numDiPerVideos=class_checkpoint['model_config']['numDynamicImages'],
                                            joinType=class_checkpoint['model_config']['joinType'],
                                            use_pretrained=True)
    
    classifier_model.load_state_dict(class_checkpoint['model_state_dict'])
    classifier_model.to(DEVICE)
    classifier_model.eval()
    checkpoint_path = None
    if args.saveCheckpoint:
        checkpoint_path = 'MaskModel_bbone={}_Dts={}_NDI-len={}-{}_AreaLoss={}_SmoothLoss={}_PreservLoss={}_AreaLoss2={}_epochs={}'.format(class_checkpoint['model_config']['modelType'],
                                                                                                                                class_checkpoint['model_config']['dataset'],
                                                                                                                                class_checkpoint['model_config']['numDynamicImages'],
                                                                                                                                class_checkpoint['model_config']['segmentLength'],
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
          numDynamicImages=class_checkpoint['model_config']['numDynamicImages'],
          checkpoint_path=checkpoint_path)
    # train(num_classes, args.num_epochs, regularizers, train_dt_loader, args.classifier, args.numDiPerVideos)

__anomaly_main__()