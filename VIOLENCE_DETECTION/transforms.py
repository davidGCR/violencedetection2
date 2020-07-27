import torchvision.transforms as transforms
import numpy as np
import torch

def hockeyTransforms(input_size, mean=None, std=None):

    if mean is None:
        mean = [0.4770381, 0.4767955, 0.4773611] #For dynamic images
        std = [0.11147115, 0.11427314, 0.11617025]
    data_transforms = {
        "train": transforms.Compose(
            [
                # transforms.Resize(input_size),
                # transforms.CenterCrop(input_size),
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
         "val": transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }
    return data_transforms

def vifTransforms(input_size,
                    train_mean=[0.5168978, 0.51586777, 0.5158742],
                    train_std=[0.12358205, 0.11996705, 0.11759791],
                    test_mean=[0.5168978, 0.51586777, 0.5158742],
                    test_std=[0.12358205, 0.11996705, 0.1175979]):
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
                transforms.Normalize(mean=train_mean, std=train_std)
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
         "val": transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=test_mean, std=test_std)
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }
    return data_transforms

def ucf2CrimeTransforms(input_size):
    mean = [0.51002795, 0.5097461, 0.5097256]
    std1 = [0.07351338, 0.07239371, 0.07159009]
    std2 = [0.07351349, 0.07239381, 0.07159019]
    data_transforms = {
        "train": transforms.Compose(
            [
                # transforms.Resize(input_size),
                # transforms.CenterCrop(input_size),
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std1) #All Train split
            ]
        ),
        "val": transforms.Compose(
            [
                # transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std1) #All Train split
                
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize(input_size),
                # transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std1) #All Train split
            ]
        )
    }
    return data_transforms

def compute_mean_std(dataloader):
    pop_mean = []
    pop_std0 = []
    pop_std1 = []
    for i, data in enumerate(dataloader, 0):
        inputs, labels, _, _ = data
        inputs = torch.squeeze(inputs, dim=1)
        numpy_image = inputs.numpy()
        # print('shape: ', numpy_image.shape)
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

if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from VIOLENCE_DETECTION.datasetsMemoryLoader import crime2localLoadData
    from VIOLENCE_DETECTION.violenceDataset import ViolenceDataset
    import torch

    X, y, numFrames = crime2localLoadData(min_frames=40)
    dataset = ViolenceDataset(dataset=X,
                            labels=y,
                            numFrames=numFrames,
                            spatial_transform=None,
                            numDynamicImagesPerVideo=1,
                            videoSegmentLength=10,
                            positionSegment='begin',
                            overlaping=0,
                            frame_skip=0,
                            skipInitialFrames=10,
                            ppType=None)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)
    pop_mean, pop_std0, pop_std1 = compute_mean_std(dataloader)
    print(pop_mean, pop_std0, pop_std1)

    

# def getDataLoaderAnomaly(x, y, numFrames, data_transform, numDiPerVideos, sequence_length, batch_size, num_workers, dataset_source, debugg_mode):
#     dataset = anomaly_dataset.AnomalyDataset( dataset=x, labels=y, numFrames=train_num_frames, spatial_transform=transforms["train"], source=dataset_source,
#              nDynamicImages=numDiPerVideos, maxNumFramesOnVideo=maxNumFramesOnVideo, videoSegmentLength=videoSegmentLength, positionSegment=positionSegment)
#     dataloader = torch.utils.data.DataLoader( dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
#     return dataloader

# def __main__():
#     print('Running main.py from transforms...')
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

#     sequence_length = 0
#     # input_size = 224
#     numDiPerVideos = 1
#     batch_size = 32
#     dataset_source = "frames"
#     data_transform = transforms.Compose([
#         transforms.ToTensor()
#     ])
#     # data_loader = getDataLoader(x, y, data_transform, numDiPerVideos, dataset_source, avgmaxDuration, interval_duration, batch_size, num_workers, debugg_mode)
#     data_loader = getDataLoaderAnomaly(x, y, numFrames, data_transform, numDiPerVideos, sequence_length, batch_size, num_workers, dataset_source, debugg_mode)
    
#     pop_mean, pop_std0, pop_std1 = compute_mean_std(data_loader)
    
#     print('pop_mean, pop_std0, pop_std1', pop_mean, pop_std0, pop_std1)

# __main__()