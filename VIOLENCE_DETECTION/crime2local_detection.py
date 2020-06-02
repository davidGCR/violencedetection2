import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
from operator import itemgetter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
import numpy as np

from constants import DEVICE
import constants
from UTIL.chooseModel import initialize_model
from UTIL.parameters import verifiParametersToTrain
from datasetsPreprocessing import crime2localgGetSplit, crime2localLoadData
from violenceDataset import ViolenceDataset
from UTIL.chooseModel import initialize_model
from UTIL.trainer import Trainer
# from UTIL.tester import Tester
from UTIL.parameters import verifiParametersToTrain
from ucfcrime2local_transforms import createTransforms
from UTIL.resultsPolicy import ResultPolicy

BASE_LR = 0.001
EPOCH_DECAY = 5 # number of epochs after which the Learning rate is decayed exponentially.
DECAY_WEIGHT = 0.1  # factor by which the learning rate is reduced.
def my_exp_lr_scheduler(optimizer, epoch, init_lr=BASE_LR, lr_decay_epoch=EPOCH_DECAY):
    """Decay learning rate by a factor of DECAY_WEIGHT every lr_decay_epoch epochs."""
    lr = init_lr * (DECAY_WEIGHT**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelType",type=str)
    parser.add_argument("--numEpochs",type=int,default=30)
    parser.add_argument("--batchSize",type=int,default=64)
    parser.add_argument("--featureExtract",type=lambda x: (str(x).lower() == 'true'), default=False, help="to fine tunning")
    parser.add_argument("--numDynamicImagesPerVideo", type=int, help="num dyn imgs")
    parser.add_argument("--joinType", type=str)
    parser.add_argument("--overlapping", type=float)
    parser.add_argument("--numWorkers", type=int, default=4)
    parser.add_argument("--videoSegmentLength", type=int, default=0)
    parser.add_argument("--positionSegment", type=str)
    parser.add_argument("--frameSkip", type=int)
    parser.add_argument("--split_type", type=str)
    parser.add_argument("--transferModel", type=str, default=None)
    parser.add_argument("--skipInitialFrames", type=int, default=0)
    parser.add_argument("--lrScheduler", type=str)
    parser.add_argument("--saveCheckpoint", type=lambda x: (str(x).lower() == 'true'), default=False)
    args = parser.parse_args()
    # train_videos_path = os.path.join(constants.PATH_UCFCRIME2LOCAL_README, 'Train_split_AD.txt')
    # test_videos_path = os.path.join(constants.PATH_UCFCRIME2LOCAL_README, 'Test_split_AD.txt')

    input_size = 224
    transforms = createTransforms(input_size)
    num_classes = 2
    shuffle = True

    cv_test_accs = []
    cv_test_losses = []
    cv_final_epochs = []

    X, y, numFrames = crime2localLoadData(min_frames=40)
    print('X={}, y={}, numFrames={}'.format(len(X), len(y), len(numFrames)))
    for i, (train_idx, test_idx) in enumerate(crime2localgGetSplit(X, y, numFrames, 5)):
        train_x = list(itemgetter(*train_idx)(X))
        train_y = list(itemgetter(*train_idx)(y))
        train_numFrames = list(itemgetter(*train_idx)(numFrames))
        test_x = list(itemgetter(*test_idx)(X))
        test_y = list(itemgetter(*test_idx)(y))
        test_numFrames = list(itemgetter(*test_idx)(numFrames))

        # print(test_x)

        image_datasets = {
            "train": ViolenceDataset(dataset=train_x,
                                    labels=train_y,
                                    numFrames=train_numFrames,
                                    spatial_transform=transforms["train"],
                                    numDynamicImagesPerVideo=args.numDynamicImagesPerVideo,
                                    videoSegmentLength=args.videoSegmentLength,
                                    positionSegment=args.positionSegment,
                                    overlaping=args.overlapping,
                                    frame_skip=args.frameSkip,
                                    skipInitialFrames=args.skipInitialFrames,
                                    preprocess_images=True),
            "test": ViolenceDataset(dataset=test_x,
                                    labels=test_y,
                                    numFrames=test_numFrames,
                                    spatial_transform=transforms["val"],
                                    numDynamicImagesPerVideo=args.numDynamicImagesPerVideo,
                                    videoSegmentLength=args.videoSegmentLength,
                                    positionSegment=args.positionSegment,
                                    overlaping=args.overlapping,
                                    frame_skip=args.frameSkip,
                                    skipInitialFrames=args.skipInitialFrames,
                                    preprocess_images=True),
        }
        dataloaders_dict = {
            "train": torch.utils.data.DataLoader(image_datasets["train"], batch_size=args.batchSize, shuffle=shuffle, num_workers=args.numWorkers),
            "test": torch.utils.data.DataLoader( image_datasets["test"], batch_size=args.batchSize, shuffle=shuffle, num_workers=args.numWorkers)
        }


    
        experimentConfig = 'UCFCRIME2LOCAL-Model-{},trainAllModel-{},TransferModel-{},lrScheduler-{},segmentLen-{},numDynIms-{},frameSkip-{},epochs-{},skipInitialFrames-{},split_type-{}, fold-{}'.format(args.modelType,
                                                                                                                                        not args.featureExtract,
                                                                                                                                        args.transferModel is not None,
                                                                                                                                        args.lrScheduler,
                                                                                                                                        args.videoSegmentLength,
                                                                                                                                        args.numDynamicImagesPerVideo,
                                                                                                                                        args.frameSkip,
                                                                                                                                        args.numEpochs,
                                                                                                                                        args.skipInitialFrames,
                                                                                                                                        args.split_type,
                                                                                                                                        i+1)
    
        
        model, _ = initialize_model(model_name=args.modelType,
                                    num_classes=2,
                                    feature_extract=args.featureExtract,
                                    numDiPerVideos=args.numDynamicImagesPerVideo,
                                    joinType=args.joinType,
                                    use_pretrained=True)
        model.to(DEVICE)
        if args.transferModel is not None:
            if DEVICE == 'cuda:0':
                model.load_state_dict(torch.load(args.transferModel), strict=False)
            else:
                model.load_state_dict(torch.load(args.transferModel, map_location=DEVICE))


        params_to_update = verifiParametersToTrain(model, args.featureExtract)
        
        optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
        
        if args.lrScheduler == 'steplr':
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
            
        criterion = nn.CrossEntropyLoss()
        
        log_dir = os.path.join(constants.PATH_RESULTS, 'UCFCRIME2LOCAL', 'tensorboard-runs', experimentConfig)
        writer = SummaryWriter(log_dir)
        print('Tensorboard logDir={}'.format(log_dir))
        
        tr = Trainer(model=model,
                    model_transfer=args.transferModel,
                    train_dataloader=dataloaders_dict['train'],
                    val_dataloader=dataloaders_dict['test'],
                    criterion=criterion,
                    optimizer=optimizer,
                    lr_scheduler=my_exp_lr_scheduler,
                    num_epochs=args.numEpochs,
                    checkpoint_path=os.path.join(constants.PATH_RESULTS,'UCFCRIME2LOCAL','checkpoints',experimentConfig))
        policy = ResultPolicy()
        for epoch in range(1, args.numEpochs + 1):
            print("Fold {} ----- Epoch {}/{}".format(i+1,epoch, args.numEpochs))
            # Train and evaluate
            epoch_loss_train, epoch_acc_train = tr.train_epoch(epoch)
            epoch_loss_val, epoch_acc_val = tr.val_epoch(epoch)
            
            if args.lrScheduler == 'steplr':
                exp_lr_scheduler.step()

            flac = policy.update(epoch_loss_train, epoch_acc_train, epoch_loss_val, epoch_acc_val, epoch)
            # print(flac, type(flac))
            if args.saveCheckpoint:
                tr.saveCheckpoint(epoch, flac)

            writer.add_scalar('training loss', epoch_loss_train, epoch)
            writer.add_scalar('validation loss', epoch_loss_val, epoch)
            writer.add_scalar('training Acc', epoch_acc_train, epoch)
            writer.add_scalar('validation Acc', epoch_acc_val, epoch)
        cv_test_accs.append(policy.getFinalTestAcc())
        cv_test_losses.append(policy.getFinalTestLoss())
        cv_final_epochs.append(policy.getFinalEpoch())
    
    print('CV Accuracies=', cv_test_accs)
    print('CV Losses=', cv_test_losses)
    print('CV Epochs=', cv_final_epochs)
    print('Test AVG Accuracy={}, Test AVG Loss={}'.format(np.average(cv_test_accs), np.average(cv_test_losses)))
        

__main__()