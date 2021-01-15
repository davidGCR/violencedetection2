# encoding: utf-8
"""
Training implementation
Author: Ian Ren
Update time: 08/11/2020
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
# import sys
# import os
import cv2
import time
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
# from read_data import ChestXrayDataSet
from sklearn.metrics import roc_auc_score
from skimage.measure import label
# from model import Densenet121_AG, Fusion_Branch

from PIL import Image
import argparse
from collections import Counter
import constants
from constants import DEVICE
from VIOLENCE_DETECTION.UTIL2 import base_dataset, load_model, transforms_dataset, plot_example
from VIOLENCE_DETECTION.violenceDataset import ViolenceDataset
from MODELS.AGCNN import Densenet121_AG, Fusion_Branch
from datasetsMemoryLoader import customize_kfold
from operator import itemgetter

#np.set_printoptions(threshold = np.nan)

# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CKPT_PATH = ''

CKPT_PATH_G = '/best_model/AG_CNN_Global_epoch_1.pkl' 
CKPT_PATH_L = '/best_model/AG_CNN_Local_epoch_2.pkl' 
CKPT_PATH_F = '/best_model/AG_CNN_Fusion_epoch_23.pkl'

N_CLASSES = 2
CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

# load with your own dataset path
DATA_DIR = '/Users/davidchoqueluqueroman/Documents/CODIGOS/AG-CNN/DATA/images' #'/media/xxxx/data/xxxx/images'
TRAIN_IMAGE_LIST = '/Users/davidchoqueluqueroman/Documents/CODIGOS/AG-CNN/labels/train_list.txt'
VAL_IMAGE_LIST = '/Users/davidchoqueluqueroman/Documents/CODIGOS/AG-CNN/labels/val_list.txt'
save_model_path = '/Users/davidchoqueluqueroman/Documents/CODIGOS/AG-CNN/model-AG-CNN/'
save_model_name = 'AG_CNN'

# learning rate
LR_G = 1e-8
LR_L = 1e-8
LR_F = 1e-3
num_epochs = 50
BATCH_SIZE = 16 #32

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.Resize((256,256)),
   transforms.CenterCrop(224),
   transforms.ToTensor(),
   normalize,
])


def Attention_gen_patchs(ori_image, fm_cuda):
    # feature map -> feature mask (using feature map to crop on the original image) -> crop -> patchs
    feature_conv = fm_cuda.data.cpu().numpy()
    size_upsample = (224, 224) 
    bz, nc, h, w = feature_conv.shape

    patchs_cuda = torch.FloatTensor().to(DEVICE)

    for i in range(0, bz):
        feature = feature_conv[i]
        cam = feature.reshape((nc, h*w))
        cam = cam.sum(axis=0)
        cam = cam.reshape(h,w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)

        heatmap_bin = binImage(cv2.resize(cam_img, size_upsample))
        heatmap_maxconn = selectMaxConnect(heatmap_bin)
        heatmap_mask = heatmap_bin * heatmap_maxconn

        ind = np.argwhere(heatmap_mask != 0)
        minh = min(ind[:,0])
        minw = min(ind[:,1])
        maxh = max(ind[:,0])
        maxw = max(ind[:,1])
        
        # to ori image 
        image = ori_image[i].numpy().reshape(224,224,3)
        image = image[int(224*0.334):int(224*0.667),int(224*0.334):int(224*0.667),:]

        image = cv2.resize(image, size_upsample)
        image_crop = image[minh:maxh,minw:maxw,:] * 256 # because image was normalized before
        image_crop = preprocess(Image.fromarray(image_crop.astype('uint8')).convert('RGB')) 

        img_variable = torch.autograd.Variable(image_crop.reshape(3,224,224).unsqueeze(0).to(DEVICE))

        patchs_cuda = torch.cat((patchs_cuda,img_variable),0)

    return patchs_cuda


def binImage(heatmap):
    _, heatmap_bin = cv2.threshold(heatmap , 0 , 255 , cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # t in the paper
    #_, heatmap_bin = cv2.threshold(heatmap , 178 , 255 , cv2.THRESH_BINARY)
    return heatmap_bin


def selectMaxConnect(heatmap):
    labeled_img, num = label(heatmap, connectivity=2, background=0, return_num=True)    
    max_label = 0
    max_num = 0
    for i in range(1, num+1):
        if np.sum(labeled_img == i) > max_num:
            max_num = np.sum(labeled_img == i)
            max_label = i
    lcc = (labeled_img == max_label)
    if max_num == 0:
       lcc = (labeled_img == -1)
    lcc = lcc + 0
    return lcc 




def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lib", type=str)
    parser.add_argument("--modelType", type=str, default="alexnet", help="model")
    parser.add_argument("--inputSize", type=int)
    parser.add_argument("--useValSplit", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--dataset", nargs='+', type=str)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--numEpochs",type=int,default=30)
    parser.add_argument("--batchSize",type=int,default=64)
    parser.add_argument("--freezeConvLayers",type=lambda x: (str(x).lower() == 'true'), default=False, help="to fine tunning")
    parser.add_argument("--numDynamicImagesPerVideo", type=int )
    parser.add_argument("--joinType", type=str)
    parser.add_argument("--foldsNumber", type=int, default=5)
    parser.add_argument("--numWorkers", type=int, default=4)
    parser.add_argument("--videoSegmentLength", type=int)
    parser.add_argument("--positionSegment", type=str)
    parser.add_argument("--splitType", type=str)
    parser.add_argument("--overlapping", type=float)
    parser.add_argument("--frameSkip", type=int, default=0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--skipInitialFrames", type=int, default=0)
    parser.add_argument("--saveCheckpoint", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--useKeyframes", type=str, default=None)
    parser.add_argument("--windowLen", type=int, default=0)
    parser.add_argument("--modelPath", type=str, default=None)
    parser.add_argument("--testDataset",type=str, default=None)
    parser.add_argument("--pretrained", type=str, default=None)

    args = parser.parse_args()
    return args

def main():
    print('********************load data********************')
    # normalize = transforms.Normalize([0.485, 0.456, 0.406],
    #                                  [0.229, 0.224, 0.225])

    # train_dataset = ChestXrayDataSet(data_dir=DATA_DIR,
    #                                 image_list_file=TRAIN_IMAGE_LIST,
    #                                 transform=transforms.Compose([
    #                                     transforms.Resize(224),
    #                                     transforms.CenterCrop(224),
    #                                     transforms.ToTensor(),
    #                                     normalize,
    #                                 ]))
    # train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,
    #                          shuffle=True, num_workers=4, pin_memory=True)
    
    # test_dataset = ChestXrayDataSet(data_dir=DATA_DIR,
    #                                 image_list_file=VAL_IMAGE_LIST,
    #                                 transform=transforms.Compose([
    #                                     transforms.Resize(256),
    #                                     transforms.CenterCrop(224),
    #                                     transforms.ToTensor(),
    #                                     normalize,
    #                                 ]))
    # test_loader = DataLoader(dataset=test_dataset, batch_size=128,
    #                          shuffle=False, num_workers=4, pin_memory=True)

    # dataset = 'hockey'
    shuffle = True
    fold=0
    # folds_number = 5
    # inputSize = 224
    # useValSplit = False
    args = build_args()
    datasetAll, labelsAll, numFramesAll, transforms = base_dataset(args.dataset[0], input_size=args.inputSize)

    for train_idx, test_idx in customize_kfold(n_splits=args.foldsNumber, dataset=args.dataset[0], X=datasetAll, y=labelsAll, shuffle=shuffle):
        fold = fold + 1

        print("**************** Fold:{}/{} ".format(fold, args.foldsNumber))
        if args.dataset[0] == 'rwf-2000':
            train_x, train_y, train_numFrames, test_x, test_y, test_numFrames, transforms = base_dataset(args.dataset[0], input_size=inputSize)
            print(len(train_x), len(train_y), len(train_numFrames), len(test_x), len(test_y), len(test_numFrames))
        else:
            train_x, train_y, test_x, test_y = None, None, None, None
            train_x = list(itemgetter(*train_idx)(datasetAll))
            train_y = list(itemgetter(*train_idx)(labelsAll))
            train_numFrames = list(itemgetter(*train_idx)(numFramesAll))
            test_x = list(itemgetter(*test_idx)(datasetAll))
            test_y = list(itemgetter(*test_idx)(labelsAll))
            test_numFrames = list(itemgetter(*test_idx)(numFramesAll))

        print('Label distribution:')
        print('Train=', Counter(train_y))
        # print('Val=', Counter(val_y))
        print('Test=', Counter(test_y))
        if args.useValSplit:
            train_x, val_x, train_numFrames, val_numFrames, train_y, val_y = train_test_split(train_x, train_numFrames, train_y, test_size=0.2, stratify=train_y, random_state=1)
            #Label distribution
            # print('Label distribution:')
            # print('Train=', Counter(train_y))
            print('Val=', Counter(val_y))
            # print('Test=', Counter(test_y))

            val_dataset = ViolenceDataset(videos=val_x,
                                            labels=val_y,
                                            numFrames=val_numFrames,
                                            spatial_transform=transforms['val'],
                                            numDynamicImagesPerVideo=args.numDynamicImagesPerVideo,
                                            videoSegmentLength=args.videoSegmentLength,
                                            positionSegment=args.positionSegment,
                                            overlaping=args.overlapping,
                                            frame_skip=args.frameSkip,
                                            skipInitialFrames=args.skipInitialFrames,
                                            ppType=None,
                                            useKeyframes=args.useKeyframes,
                                            windowLen=args.windowLen,
                                            dataset=args.dataset[0])

        train_dataset = ViolenceDataset(videos=train_x,
                                        labels=train_y,
                                        numFrames=train_numFrames,
                                        spatial_transform=transforms['train'],
                                        numDynamicImagesPerVideo=args.numDynamicImagesPerVideo,
                                        videoSegmentLength=args.videoSegmentLength,
                                        positionSegment=args.positionSegment,
                                        overlaping=args.overlapping,
                                        frame_skip=args.frameSkip,
                                        skipInitialFrames=args.skipInitialFrames,
                                        ppType=None,
                                        useKeyframes=args.useKeyframes,
                                        windowLen=args.windowLen,
                                        dataset=args.dataset[0])

        test_dataset = ViolenceDataset(videos=test_x,
                                        labels=test_y,
                                        numFrames=test_numFrames,
                                        spatial_transform=transforms['val'],
                                        numDynamicImagesPerVideo=args.numDynamicImagesPerVideo,
                                        videoSegmentLength=args.videoSegmentLength,
                                        positionSegment=args.positionSegment,
                                        overlaping=args.overlapping,
                                        frame_skip=args.frameSkip,
                                        skipInitialFrames=args.skipInitialFrames,
                                        ppType=None,
                                        useKeyframes=args.useKeyframes,
                                        windowLen=args.windowLen,
                                        dataset=args.dataset[0])


        if not args.useValSplit:
            val_dataset = test_dataset
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=True, num_workers=args.numWorkers)
        test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchSize, shuffle=True, num_workers=args.numWorkers)

        print('********************load data succeed!********************')


        print('********************load model********************')
        # initialize and load the model
        Global_Branch_model = Densenet121_AG(pretrained = False, num_classes = N_CLASSES).to(DEVICE)
        Local_Branch_model = Densenet121_AG(pretrained = False, num_classes = N_CLASSES).to(DEVICE)
        Fusion_Branch_model = Fusion_Branch(input_size = 2048, output_size = N_CLASSES).to(DEVICE)

        if os.path.isfile(CKPT_PATH):
            print("=> loading checkpoint")
            checkpoint = torch.load(CKPT_PATH)
            # to load state
            # Code modified from torchvision densenet source for loading from pre .4 densenet weights.
            state_dict = checkpoint['state_dict']
            remove_data_parallel = True # Change if you don't want to use nn.DataParallel(model)

            pattern = re.compile(
                r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
            for key in list(state_dict.keys()):
                ori_key =  key
                key = key.replace('densenet121.','')
                #print('key',key)
                match = pattern.match(key)
                new_key = match.group(1) + match.group(2) if match else key
                new_key = new_key[7:] if remove_data_parallel else new_key
                #print('new_key',new_key)
                if '.0.' in new_key:
                    new_key = new_key.replace('0.','')
                state_dict[new_key] = state_dict[ori_key]
                # Delete old key only if modified.
                if match or remove_data_parallel: 
                    del state_dict[ori_key]
            
            Global_Branch_model.load_state_dict(state_dict)
            Local_Branch_model.load_state_dict(state_dict)
            print("=> loaded baseline checkpoint")
            
        else:
            print("=> no checkpoint found")

        if os.path.isfile(CKPT_PATH_G):
            checkpoint = torch.load(CKPT_PATH_G)
            Global_Branch_model.load_state_dict(checkpoint)
            print("=> loaded Global_Branch_model checkpoint")

        if os.path.isfile(CKPT_PATH_L):
            checkpoint = torch.load(CKPT_PATH_L)
            Local_Branch_model.load_state_dict(checkpoint)
            print("=> loaded Local_Branch_model checkpoint")

        if os.path.isfile(CKPT_PATH_F):
            checkpoint = torch.load(CKPT_PATH_F)
            Fusion_Branch_model.load_state_dict(checkpoint)
            print("=> loaded Fusion_Branch_model checkpoint")

        cudnn.benchmark = True
        # criterion = nn.BCELoss()
        criterion = nn.CrossEntropyLoss()
        optimizer_global = optim.Adam(Global_Branch_model.parameters(), lr=LR_G, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        lr_scheduler_global = lr_scheduler.StepLR(optimizer_global , step_size = 10, gamma = 1)
        
        optimizer_local = optim.Adam(Local_Branch_model.parameters(), lr=LR_L, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        lr_scheduler_local = lr_scheduler.StepLR(optimizer_local , step_size = 10, gamma = 1)
        
        optimizer_fusion = optim.Adam(Fusion_Branch_model.parameters(), lr=LR_F, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        lr_scheduler_fusion = lr_scheduler.StepLR(optimizer_fusion , step_size = 15, gamma = 0.1)
        print('********************load model succeed!********************')

        print('********************begin training!********************')
        for epoch in range(num_epochs):
            since = time.time()
            print('Epoch {}/{}'.format(epoch , num_epochs - 1))
            print('-' * 10)
            #set the mode of model
            lr_scheduler_global.step()  #about lr and gamma
            lr_scheduler_local.step() 
            lr_scheduler_fusion.step() 
            Global_Branch_model.train()  #set model to training mode
            Local_Branch_model.train()
            Fusion_Branch_model.train()

            running_loss = 0.0
            #Iterate over data
            # for i, (input, target) in enumerate(train_loader):
            #     print(type(input))

            for i, (input, target) in enumerate(train_loader):
                # input_var = torch.autograd.Variable(input.to(DEVICE))
                # target_var = torch.autograd.Variable(target.to(DEVICE))

                (input, vid_name, dynamicImages, bboxes) = input
                batch_size, timesteps, C, H, W = input.size()
                input = input.view(batch_size * timesteps, C, H, W)
                input_var = input.to(DEVICE)
                target_var = target.to(DEVICE)

                optimizer_global.zero_grad()
                optimizer_local.zero_grad()
                optimizer_fusion.zero_grad()

                # compute output
                output_global, fm_global, pool_global = Global_Branch_model(input_var)
                
                patchs_var = Attention_gen_patchs(input,fm_global)

                output_local, _, pool_local = Local_Branch_model(patchs_var)
                #print(fusion_var.shape)
                output_fusion = Fusion_Branch_model(pool_global, pool_local)
                #

                # loss
                loss1 = criterion(output_global, target_var)
                loss2 = criterion(output_local, target_var)
                loss3 = criterion(output_fusion, target_var)
                #

                loss = loss1*0.8 + loss2*0.1 + loss3*0.1 

                if (i%500) == 0: 
                    print('step: {} totalloss: {loss:.3f} loss1: {loss1:.3f} loss2: {loss2:.3f} loss3: {loss3:.3f}'.format(i, loss = loss, loss1 = loss1, loss2 = loss2, loss3 = loss3))

                loss.backward() 
                optimizer_global.step()  
                optimizer_local.step()
                optimizer_fusion.step()

                #print(loss.data.item())
                running_loss += loss.data.item()
                #break
                '''
                if i == 40:
                    print('break')
                    break
                '''

            epoch_loss = float(running_loss) / float(i)
            print(' Epoch over  Loss: {:.5f}'.format(epoch_loss))

            print('*******testing!*********')
            test(Global_Branch_model, Local_Branch_model, Fusion_Branch_model,test_loader)
            #break

            #save
            if epoch % 1 == 0:
                save_path = save_model_path
                torch.save(Global_Branch_model.state_dict(), save_path+save_model_name+'_Global'+'_epoch_'+str(epoch)+'.pkl')
                print('Global_Branch_model already save!')
                torch.save(Local_Branch_model.state_dict(), save_path+save_model_name+'_Local'+'_epoch_'+str(epoch)+'.pkl')
                print('Local_Branch_model already save!')
                torch.save(Fusion_Branch_model.state_dict(), save_path+save_model_name+'_Fusion'+'_epoch_'+str(epoch)+'.pkl')            
                print('Fusion_Branch_model already save!')

            time_elapsed = time.time() - since
            print('Training one epoch complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60 , time_elapsed % 60))
    

def test(model_global, model_local, model_fusion, test_loader):

    # initialize the ground truth and output tensor
    gt = torch.FloatTensor().to(DEVICE)
    pred_global = torch.FloatTensor().to(DEVICE)
    pred_local = torch.FloatTensor().to(DEVICE)
    pred_fusion = torch.FloatTensor().to(DEVICE)

    # switch to evaluate mode
    model_global.eval()
    model_local.eval()
    model_fusion.eval()
    cudnn.benchmark = True

    for i, (inp, target) in enumerate(test_loader):
        with torch.no_grad():
            if i % 2000 == 0:
                print('testing process:',i)
            target = target.to(DEVICE)
            gt = torch.cat((gt, target), 0)
            # input_var = torch.autograd.Variable(inp.to(DEVICE))
            (inp, vid_name, dynamicImages, bboxes) = inp
            batch_size, timesteps, C, H, W = inp.size()
            inp = inp.view(batch_size * timesteps, C, H, W)
            input_var = inp.to(DEVICE)
            # target_var = target.to(DEVICE)

            #output = model_global(input_var)

            output_global, fm_global, pool_global = model_global(input_var)
            
            patchs_var = Attention_gen_patchs(inp,fm_global)

            output_local, _, pool_local = model_local(patchs_var)

            output_fusion = model_fusion(pool_global,pool_local)

            pred_global = torch.cat((pred_global, output_global.data), 0)
            pred_local = torch.cat((pred_local, output_local.data), 0)
            pred_fusion = torch.cat((pred_fusion, output_fusion.data), 0)
            
    AUROCs_g = compute_AUCs(gt, pred_global)
    AUROC_avg = np.array(AUROCs_g).mean()
    print('Global branch: The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
    for i in range(N_CLASSES):
        print('The AUROC of {} is {}'.format(CLASS_NAMES[i], AUROCs_g[i]))

    AUROCs_l = compute_AUCs(gt, pred_local)
    AUROC_avg = np.array(AUROCs_l).mean()
    print('\n')
    print('Local branch: The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
    for i in range(N_CLASSES):
        print('The AUROC of {} is {}'.format(CLASS_NAMES[i], AUROCs_l[i]))

    AUROCs_f = compute_AUCs(gt, pred_fusion)
    AUROC_avg = np.array(AUROCs_f).mean()
    print('\n')
    print('Fusion branch: The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
    for i in range(N_CLASSES):
        print('The AUROC of {} is {}'.format(CLASS_NAMES[i], AUROCs_f[i]))


def compute_AUCs(gt, pred):
    """Computes Area Under the Curve (AUC) from prediction scores.

    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.

    Returns:
        List of AUROCs of all classes.
    """
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(N_CLASSES):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return AUROCs

if __name__ == '__main__':
    
    main()