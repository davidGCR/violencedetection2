import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import constants
import ANOMALYCRIME.anomaly_dataset as anomaly_dataset
import os
import torch
import glob

def createTransforms(input_size):
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        "train": transforms.Compose(
            [
                # transforms.Resize(input_size),
                # transforms.CenterCrop(input_size),
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.49237782, 0.49160805, 0.48998737], [0.11053326, 0.11088469, 0.11275752] )
                # [0.49237782 0.49160805 0.48998737] [0.11053326 0.11088469 0.11275752] [0.11053331 0.11088473 0.11275754]
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.49237782, 0.49160805, 0.48998737], [0.11053326, 0.11088469, 0.11275752] )
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize(input_size),
                # transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.49237782, 0.49160805, 0.48998737], [0.11053326, 0.11088469, 0.11275752] )
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    }
    return data_transforms

def compute_mean_std(dataloader):
    pop_mean = []
    pop_std0 = []
    pop_std1 = []
    for i, data in tqdm(enumerate(dataloader, 0)):
        # shape (batch_size, 3, height, width)
        inputs, labels, _ = data
        numpy_image = inputs.numpy()
        
        # shape (3,)
        batch_mean = np.mean(numpy_image, axis=(0,2,3))
        batch_std0 = np.std(numpy_image, axis=(0,2,3))
        batch_std1 = np.std(numpy_image, axis=(0,2,3), ddof=1)
        
        pop_mean.append(batch_mean)
        pop_std0.append(batch_std0)
        pop_std1.append(batch_std1)

    # shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
    pop_mean = np.array(pop_mean).mean(axis=0)
    pop_std0 = np.array(pop_std0).mean(axis=0)
    pop_std1 = np.array(pop_std1).mean(axis=0)
    return pop_mean, pop_std0, pop_std1

def getDataLoaderAnomaly(x, y, numFrames, data_transform, numDiPerVideos, maxNumFramesOnVideo, batch_size, num_workers, dataset_source, debugg_mode):
    dataset = anomaly_dataset.AnomalyDataset( dataset=x, labels=y, numFrames=numFrames, spatial_transform=data_transform, source=dataset_source,
             nDynamicImages=numDiPerVideos, debugg_mode=debugg_mode, maxNumFramesOnVideo=maxNumFramesOnVideo)
    dataloader = torch.utils.data.DataLoader( dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader

# def __main__():
#     print('Main of Transforms Anomaly...')
#     train_videos_path = os.path.join(constants.PATH_UCFCRIME2LOCAL_README, 'Train_split_AD.txt')
#     test_videos_path = os.path.join(constants.PATH_UCFCRIME2LOCAL_README, 'Test_split_AD.txt')
    
#     train_names, train_labels, NumFrames_train, test_names, test_labes, NumFrames_test = anomaly_dataset.train_test_videos(train_videos_path, test_videos_path, constants.PATH_UCFCRIME2LOCAL_FRAMES_REDUCED)
#     x = train_names + test_names
#     y = train_labels + test_labes
#     numFrames = NumFrames_train + NumFrames_test
#     # hockey_path_violence = "/media/david/datos/Violence DATA/HockeyFights/frames/violence"
#     # hockey_path_noviolence = "/media/david/datos/Violence DATA/HockeyFights/frames/nonviolence"
#     # x, y, numFramesAll = createDataset(hockey_path_violence, hockey_path_noviolence)  #ordered

#     debugg_mode = False
#     avgmaxDuration = 1.66
#     interval_duration = 0.3
#     num_workers = 4

#     maxNumFramesOnVideo = 0
#     # input_size = 224
#     numDiPerVideos = 1
#     batch_size = 32
#     dataset_source = "frames"
#     data_transform = transforms.Compose([
#         transforms.ToTensor()
#     ])
#     # data_loader = getDataLoader(x, y, data_transform, numDiPerVideos, dataset_source, avgmaxDuration, interval_duration, batch_size, num_workers, debugg_mode)
#     data_loader = getDataLoaderAnomaly(x, y, numFrames, data_transform, numDiPerVideos, maxNumFramesOnVideo, batch_size, num_workers, dataset_source, debugg_mode)
    
#     pop_mean, pop_std0, pop_std1 = compute_mean_std(data_loader)
    
#     print('pop_mean, pop_std0, pop_std1', pop_mean, pop_std0, pop_std1)

# __main__()