import initializeDataset
import constants
import violenceDataset
import argparse
import torch
import transforms
import Saliency.saliencyTrainer as saliencyTrainer
import os

def init(batch_size, num_workers, interval_duration, data_transforms, dataset_source, debugg_mode, numDiPerVideos, avgmaxDuration, shuffle=True):
    datasetAll, labelsAll, numFramesAll = initializeDataset.createDataset(constants.PATH_HOCKEY_FRAMES_VIOLENCE, constants.PATH_HOCKEY_FRAMES_NON_VIOLENCE) #ordered
    # combined = list(zip(datasetAll, labelsAll, numFramesAll))
    # random.shuffle(combined)
    # datasetAll[:], labelsAll[:], numFramesAll[:] = zip(*combined) 
    # print(len(datasetAll), len(labelsAll), len(numFramesAll))
    train_x, train_y, _, _ = initializeDataset.train_test_split_saliency(datasetAll,labelsAll, numFramesAll)
    print(len(train_x), len(train_y), len(numFramesAll))
    image_datasets = {
        "train": violenceDataset.ViolenceDataset( dataset=train_x, labels=train_y, spatial_transform=data_transforms["train"], source=dataset_source,
                interval_duration=interval_duration,difference=3, maxDuration=avgmaxDuration, nDynamicImages=numDiPerVideos, debugg_mode=debugg_mode, ),
    }
    dataloaders_dict = {
        "train": torch.utils.data.DataLoader( image_datasets["train"], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, ),
    }
    return image_datasets, dataloaders_dict

def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batchSize", type=int, default=8)
    parser.add_argument("--numEpochs", type=int, default=10)
    parser.add_argument("--numWorkers", type=int, default=4)
    parser.add_argument("--areaL", type=float, default=None)
    parser.add_argument("--smoothL", type=float, default=None)
    parser.add_argument("--preserverL", type=float, default=None)
    parser.add_argument("--areaPowerL", type=float, default=None)
    parser.add_argument("--saliencyModelFolder",type=str, default=constants.PATH_SALIENCY_MODELS)
    parser.add_argument("--blackBoxFile",type=str) #areaL-9.0_smoothL-0.3_epochs-20
    # parser.add_argument("--areaL", type=float, default=8)
    # parser.add_argument("--smoothL", type=float, default=0.5)
    # parser.add_argument("--preserverL", type=float, default=0.3)
    # parser.add_argument("--areaPowerL", type=float, default=0.3)
    # parser.add_argument("--checkpointInfo",type=str)
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    interval_duration = 0
    avgmaxDuration = 0
    numDiPerVideos = 1
    input_size = 224
    data_transforms = transforms.createTransforms(input_size)
    dataset_source = 'frames'
    debugg_mode = False
    batch_size = args.batchSize
    num_workers = args.numWorkers
    num_epochs = args.numEpochs
    black_box_file = args.blackBoxFile
    saliency_model_folder = args.saliencyModelFolder
    num_classes = 2
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

    checkpoint_path = os.path.join(saliency_model_folder, 'saliency_model' + checkpoint_info + '_epochs-' + str(num_epochs) + '.tar')
    
    image_datasets, dataloaders_dict = init(batch_size, num_workers, interval_duration, data_transforms, dataset_source, debugg_mode, numDiPerVideos, avgmaxDuration)
    saliencyTrainer.train(num_classes, num_epochs, regularizers, device, checkpoint_path, dataloaders_dict, black_box_file)

__main__()