import sys
sys.path.insert(1, "/media/david/datos/PAPERS-SOURCE_CODE/MyCode")
# import initializeDataset
import constants
import violenceDataset
import argparse
import torch
import SALIENCY.saliencyTrainer as saliencyTrainer
import os
# import anomaly_dataset
from MODELS.ResNet import *
import ANOMALYCRIME.anomaly_initializeDataset as anomaly_initializeDataset
import transforms_anomaly

# def init(batch_size, num_workers, maxNumFramesOnVideo, transforms, numDiPerVideos, avgmaxDuration, dataset_source, shuffle, videoSegmentLength, positionSegment):
#     train_names, train_labels, NumFrames_train, test_names, test_labes, NumFrames_test = anomaly_dataset.train_test_videos(constants.ANOMALY_PATH_TRAIN_SPLIT, constants.ANOMALY_PATH_TEST_SPLIT, constants.PATH_UCFCRIME2LOCAL_FRAMES_REDUCED)
#     print(len(train_names), len(test_names))
#     image_datasets = {
#         "train": anomaly_dataset.AnomalyDataset( dataset=train_names, labels=train_labels, numFrames=NumFrames_train, spatial_transform=transforms["train"], source=dataset_source,
#              nDynamicImages=numDiPerVideos, maxNumFramesOnVideo=maxNumFramesOnVideo, videoSegmentLength=videoSegmentLength, positionSegment=positionSegment),
#     }
#     dataloaders_dict = {
#         "train": torch.utils.data.DataLoader( image_datasets["train"], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, ),
#     }
#     return image_datasets, dataloaders_dict

def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batchSize", type=int, default=8)
    parser.add_argument("--numEpochs", type=int, default=10)
    parser.add_argument("--numWorkers", type=int, default=4)
    parser.add_argument("--areaL", type=float, default=None)
    parser.add_argument("--smoothL", type=float, default=None)
    parser.add_argument("--preserverL", type=float, default=None)
    parser.add_argument("--areaPowerL", type=float, default=None)
    parser.add_argument("--saliencyCheckout",type=str, default=constants.ANOMALY_PATH_SALIENCY_MODELS)
    parser.add_argument("--blackBoxFile", type=str)  #areaL-9.0_smoothL-0.3_epochs-20
    parser.add_argument("--maxNumFramesOnVideo", type=int)
    parser.add_argument("--videoSegmentLength", type=int)
    parser.add_argument("--positionSegment",type=str, default='random')
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # interval_duration = 0
    # avgmaxDuration = 0
    numDiPerVideos = 1
    input_size = 224
    # data_transforms = transforms.createTransforms(input_size)
    dataset_source = 'frames'
    # debugg_mode = False
    batch_size = args.batchSize
    num_workers = args.numWorkers
    num_epochs = args.numEpochs
    black_box_file = args.blackBoxFile
    saliency_model_folder = args.saliencyCheckout
    num_classes = 2
    shuffle = True
    maxNumFramesOnVideo = args.maxNumFramesOnVideo
    videoSegmentLength = args.videoSegmentLength
    positionSegment = args.positionSegment

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
    
    # image_datasets, dataloaders_dict = init(batch_size, num_workers, maxNumFramesOnVideo, data_transforms,
    #             numDiPerVideos, avgmaxDuration, dataset_source, shuffle, videoSegmentLength, positionSegment)
    path_dataset = constants.PATH_UCFCRIME2LOCAL_FRAMES_REDUCED
    train_videos_path = os.path.join(constants.PATH_UCFCRIME2LOCAL_README, 'Train_split_AD.txt')
    test_videos_path = os.path.join(constants.PATH_UCFCRIME2LOCAL_README, 'Test_split_AD.txt')
    transforms = transforms_anomaly.createTransforms(input_size)
    dataloaders_dict, test_names = anomaly_initializeDataset.initialize_final_anomaly_dataset(path_dataset, train_videos_path, test_videos_path, dataset_source, batch_size, num_workers,
                                                            numDiPerVideos, transforms, maxNumFramesOnVideo, videoSegmentLength, positionSegment, shuffle)

    model_name = 'resnet18'
    joinType = 'tempMaxPool'
    feature_extract = True
    black_box_model = ViolenceModelResNet(num_classes, numDiPerVideos, model_name, joinType, feature_extract)
    saliencyTrainer.train(black_box_model, num_classes, num_epochs, regularizers, device, checkpoint_path, dataloaders_dict, black_box_file)

__main__()