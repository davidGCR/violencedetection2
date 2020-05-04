import sys
sys.path.insert(1, '/media/david/datos/PAPERS-SOURCE_CODE/violencedetection')
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from torch.autograd import Variable
from scipy import misc
# import SALIENCY.saliencyModel as saliencyModel
# from saliencyModel import build_saliency_model
# from .saliencyModel  import build_saliency_model
import SALIENCY.saliencyModel
from tqdm import tqdm
from operator import itemgetter
# import transforms
from loss import Loss
from torch.optim import lr_scheduler
import os
# import util
import argparse
import constants
import ANOMALYCRIME.transforms_anomaly as transforms_anomaly
import ANOMALYCRIME.anomalyInitializeDataset as anomaly_initializeDataset
# from Models import AlexNet

def save_checkpoint(state, filename='sal.pth.tar'):
    print('save in: ',filename)
    torch.save(state, filename)

def load_checkpoint(net,optimizer,filename='small.pth.tar'):
    checkpoint = torch.load(filename)
    net.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return net,optimizer




def train(num_classes, num_epochs, regularizers, device, checkpoint_path, dataloaders_dict, black_box_file, numDynamicImages):
    saliency_m = SALIENCY.saliencyModel.build_saliency_model(num_classes=num_classes)
    saliency_m = saliency_m.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(saliency_m.parameters())           
    black_box_model = torch.load(black_box_file)
    black_box_model = black_box_model.cuda()
    black_box_model.inferenceMode()

    loss_func = Loss(num_classes=num_classes, regularizers=regularizers)
    best_loss = 1000.0
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        print("----- Epoch {}/{}".format(epoch+1, num_epochs))
        running_loss = 0.0
        running_loss_train = 0.0
        # running_corrects = 0.0
        
        for i, data in tqdm(enumerate(dataloaders_dict['train'], 0)):
            # get the inputs
            inputs, labels, video_name, bbox_segments = data #dataset load [bs,ndi,c,w,h]
            # print('dataset element: ',inputs_r.shape) #torch.Size([8, 1, 3, 224, 224])
            if numDynamicImages > 1:
                inputs = inputs.permute(1, 0, 2, 3, 4)
                inputs = torch.squeeze(inputs, 0) #get one di [bs,c,w,h]
            # print('inputs shape:',inputs.shape)
            # wrap them in Variable
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

            # zero the parameter gradients
            optimizer.zero_grad()

            mask, out = saliency_m(inputs, labels)
            # print('mask shape:', mask.shape)
            # print('inputs shape:',inputs.shape)
            # print('labels shape:', labels.shape)
            # print(labels)

            # inputs_r = Variable(inputs_r.cuda())
            loss = loss_func.get(mask,inputs,labels,black_box_model)
            # running_loss += loss.data[0]
            running_loss += loss.item()
            running_loss_train += loss.item()*inputs.size(0)
            # if(i%10 == 0):
            #     print('Epoch = %f , Loss = %f '%(epoch+1 , running_loss/(batch_size*(i+1))) )
        
            loss.backward()
            optimizer.step()
        # exp_lr_scheduler.step(running_loss)

        epoch_loss = running_loss / len(dataloaders_dict["train"].dataset)
        epoch_loss_train = running_loss_train / len(dataloaders_dict["train"].dataset)
        print("{} RawLoss: {:.4f} Loss: {:.4f}".format('train', epoch_loss, epoch_loss_train))

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            # self.best_model_wts = copy.deepcopy(self.model.state_dict())
            print('Saving entire saliency model...')
            save_checkpoint(saliency_m,checkpoint_path)

def __anomaly_main__():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batchSize", type=int, default=8)
    parser.add_argument("--numEpochs", type=int, default=10)
    parser.add_argument("--numWorkers", type=int, default=4)
    parser.add_argument("--areaL", type=float, default=None)
    parser.add_argument("--smoothL", type=float, default=None)
    parser.add_argument("--preserverL", type=float, default=None)
    parser.add_argument("--areaPowerL", type=float, default=None)
    parser.add_argument("--blackBoxFile", type=str)  #areaL-9.0_smoothL-0.3_epochs-20
    parser.add_argument("--videoSegmentLength", type=int, default=0)
    parser.add_argument("--positionSegment", type=str, default='random')
    parser.add_argument("--maxNumFramesOnVideo", type=int, default=0)
    parser.add_argument("--shuffle", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--numDiPerVideos", type=int)
    
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    numDiPerVideos = args.numDiPerVideos
    input_size = 224
    transforms_dataset = transforms_anomaly.createTransforms(input_size)
    batch_size = args.batchSize
    num_workers = args.numWorkers
    num_epochs = args.numEpochs
    black_box_file = args.blackBoxFile
    saliency_model_folder = constants.ANOMALY_PATH_SALIENCY_MODELS
    num_classes = 2
    videoSegmentLength = args.videoSegmentLength
    positionSegment = args.positionSegment
    maxNumFramesOnVideo = args.maxNumFramesOnVideo
    shuffle = args.shuffle
    # regularizers = {'area_loss_coef': args.areaL, 'smoothness_loss_coef': args.smoothL, 'preserver_loss_coef': args.preserverL, 'area_loss_power': args.areaPowerL}
    # checkpoint_info = args.checkpointInfo
    checkpoint_info = ''
    areaL, smoothL, preserverL, areaPowerL = None,None,None,None
    
    if args.areaL == None:
        areaL = 8
    else:
        areaL = args.areaL
        checkpoint_info += '_areaL-'+str(args.areaL)
    
    if args.smoothL == None:
        smoothL = 0.5
    else:
        smoothL = args.smoothL
        checkpoint_info += '_smoothL-' + str(args.smoothL)
    
    if args.preserverL == None:
        preserverL = 0.3
    else:
        preserverL = args.preserverL
        checkpoint_info += '_preserverL-' + str(args.preserverL)
    
    if args.areaPowerL == None:
        areaPowerL = 0.3
    else:
        areaPowerL = args.areaPowerL
        checkpoint_info += '_areaPowerL-' + str(args.areaPowerL)

    print('areaL, smoothL, preserverL, _areaPowerL=',areaL, smoothL, preserverL, areaPowerL)
    
    regularizers = {'area_loss_coef': areaL, 'smoothness_loss_coef': smoothL, 'preserver_loss_coef': preserverL, 'area_loss_power': areaPowerL}

    path_dataset = constants.PATH_UCFCRIME2LOCAL_FRAMES_REDUCED
    train_videos_path = os.path.join(constants.PATH_UCFCRIME2LOCAL_README, 'Train_split_AD.txt')
    test_videos_path = os.path.join(constants.PATH_UCFCRIME2LOCAL_README, 'Test_split_AD.txt')
    dataloaders_dict, test_names = anomaly_initializeDataset.initialize_final_anomaly_dataset(path_dataset, train_videos_path, test_videos_path,
                                    batch_size, num_workers, numDiPerVideos, transforms_dataset, maxNumFramesOnVideo, videoSegmentLength, positionSegment, shuffle,0)
    
    checkpoint_path = os.path.join(saliency_model_folder,'mask_model_'+str(videoSegmentLength)+'_frames_di_'+checkpoint_info + '_epochs-' + str(num_epochs) + '.tar')
    
    # image_datasets, dataloaders_dict = init_anomaly(batch_size, num_workers, maxNumFramesOnVideo, data_transforms, numDiPerVideos, avgmaxDuration, dataset_source, shuffle, videoSegmentLength, positionSegment)

    # image_datasets, dataloaders_dict = init(batch_size, num_workers, interval_duration, data_transforms, dataset_source, debugg_mode, numDiPerVideos, avgmaxDuration)
    train(num_classes, num_epochs, regularizers, device, checkpoint_path, dataloaders_dict, black_box_file, numDiPerVideos)

__anomaly_main__()