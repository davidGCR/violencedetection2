import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import constants
from constants import DEVICE
from VIOLENCE_DETECTION.transforms import hockeyTransforms
from VIOLENCE_DETECTION.datasetsMemoryLoader import hockeyLoadData
from UTIL.kfolds import k_folds
from operator import itemgetter
from VIOLENCE_DETECTION.rgbDataset import RGBDataset
from MODELS.ViolenceModels import ResNetRGB
from UTIL.parameters import verifiParametersToTrain
from UTIL.earlyStop import EarlyStopping
# from UTIL.args import Arguments
import torch
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import numpy as np
import argparse


def train_model(model, dataloaders, criterion, optimizer, num_epochs, patience, fold, path, model_config):
    since = time.time()
    val_acc_history = []
    # best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=path, model_config=model_config)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        last_train_loss = np.inf
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in tqdm(dataloaders[phase]):
                
                inputs, labels = data
                # print('inputs=', inputs.size(), type(inputs))
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                # zero the parameter gradients
                optimizer.zero_grad()
                batch_size = inputs.size()[0]
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)


            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            if phase == 'val':
                early_stopping(epoch_loss, epoch_acc, last_train_loss, epoch, fold, model)
            else:
                last_train_loss = epoch_loss
        if early_stopping.early_stop:
            print("Early stopping")
            break
            

            # deep copy the model
            # if phase == 'val' and epoch_acc > best_acc:
            #     best_acc = epoch_acc
            #     # best_model_wts = copy.deepcopy(model.state_dict())
            # if phase == 'val':
            #     val_acc_history.append(epoch_acc)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    # model.load_state_dict(best_model_wts)
    
    return model, early_stopping.best_acc, early_stopping.val_loss_min, early_stopping.best_epoch

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
    elif dataset == 'VIF':
    elif dataset == 'HOCKEY':
        mytransfroms = hockeyTransforms(224)
        datasetAll, labelsAll, numFramesAll = hockeyLoadData()
        train_x, train_y, train_numFrames, test_x, test_y, test_numFrames = hockeyTrainTestSplit('train-test-' + str(fold), datasetAll, labelsAll, numFramesAll)
    return test_x, test_y, test_numFrames, mytransfroms

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--model", type=str, default='resnet50')
    parser.add_argument("--frameIdx", type=int, default=14)
    parser.add_argument("--numEpochs", type=int, default=25)
    parser.add_argument("--numWorkers", type=int, default=4)
    parser.add_argument("--batchSize", type=int, default=8)
    parser.add_argument("--featureExtract", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--saveCheckpoint", type=lambda x: (str(x).lower() == 'true'), default=False)
    args = parser.parse_args()

    folds_number = 5
    fold = 0
    # frame_idx = 14
    input_size = 224
    # batch_size = 8
    # num_workers = 4
    # numEpochs = 25
    if args.dataset == 'hockey':
        datasetAll, labelsAll, numFramesAll = hockeyLoadData()
        transforms = hockeyTransforms(input_size, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # feature_extract = False
    # model_name = 'resnet50'
    patience = 5
    cv_test_accs = []
    cv_test_losses = []
    cv_final_epochs = []
    checkpoint_path = None
    config = None

    
    for train_idx, test_idx in k_folds(n_splits=folds_number, subjects=len(datasetAll), splits_folder=constants.PATH_HOCKEY_README):
        fold = fold + 1
        print("**************** Fold:{}/{} ".format(fold, folds_number))
        train_x, train_y, test_x, test_y = None, None, None, None
        train_x = list(itemgetter(*train_idx)(datasetAll))
        train_y = list(itemgetter(*train_idx)(labelsAll))
        train_numFrames = list(itemgetter(*train_idx)(numFramesAll))
        test_x = list(itemgetter(*test_idx)(datasetAll))
        test_y = list(itemgetter(*test_idx)(labelsAll))
        test_numFrames = list(itemgetter(*test_idx)(numFramesAll))

        train_dataset = RGBDataset(dataset=train_x,
                                labels=train_y,
                                numFrames=train_numFrames,
                                spatial_transform=transforms['train'],
                                frame_idx=args.frameIdx)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=True, num_workers=args.numWorkers)

        test_dataset = RGBDataset(dataset=test_x,
                                labels=test_y,
                                numFrames=test_numFrames,
                                spatial_transform=transforms['test'],
                                frame_idx=args.frameIdx)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batchSize, shuffle=True, num_workers=args.numWorkers)

        model = ResNetRGB(num_classes=2, model_name=args.model, feature_extract=args.featureExtract)
        model = model.build_model()
        model.to(DEVICE)
        params_to_update = verifiParametersToTrain(model, args.featureExtract)
        optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        criterion = nn.CrossEntropyLoss()

        dataloaders = {'train': train_dataloader, 'val': test_dataloader}
        if args.saveCheckpoint:
            config = {
                'dataset':args.dataset,
                'model': args.model,
                'frameIdx': args.frameIdx,
                'numEpochs': args.numEpochs,
                'featureExtract': args.featureExtract
            }
            ss = '_'.join("{!s}={!r}".format(key, val) for (key, val) in config.items())
            ss = ss+'_fold='+str(fold)
            # print(ss)
            checkpoint_path = os.path.join(constants.PATH_RESULTS, 'HOCKEY', 'checkpoints', 'RGBCNN-'+ss)

        model, best_acc, val_loss_min, best_epoch = train_model(model,
                                                                dataloaders,
                                                                criterion,
                                                                optimizer,
                                                                num_epochs=args.numEpochs,
                                                                patience=patience,
                                                                fold=fold,
                                                                path=checkpoint_path,
                                                                model_config=config)
        cv_test_accs.append(best_acc.item())
        cv_test_losses.append(val_loss_min)
        cv_final_epochs.append(best_epoch)
    print('CV Accuracies=', cv_test_accs)
    print('CV Losses=', cv_test_losses)
    print('CV Epochs=', cv_final_epochs)
    print('Test AVG Accuracy={}, Test AVG Loss={}'.format(np.average(cv_test_accs), np.average(cv_test_losses)))
    print("Accuracy: %0.3f (+/- %0.3f), Losses: %0.3f" % (np.array(cv_test_accs).mean(), np.array(cv_test_accs).std() * 2, np.array(cv_test_losses).mean()))
        



            



if __name__ == "__main__":
    main()