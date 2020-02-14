
# import sys
# sys.path.insert(1, '/media/david/datos/PAPERS-SOURCE_CODE/MyCode')
import ANOMALYCRIME.anomalyDataset as anomalyDataset
import ANOMALYCRIME.anomalyOnlineDataset as anomalyOnlineDataset
import ANOMALYCRIME.anomalyDatasetAumented as anomalyDatasetAumented
import ANOMALYCRIME.datasetUtils as datasetUtils
import ANOMALYCRIME.denseDataset as denseDataset
# import anomaly_dataset
import util
import random
import os
import constants
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

def dataloader_train_val_test_dense_dataset(path_dataset, train_videos_path, test_videos_path, batch_size, num_workers, numDiPerVideos,
                         transforms_t, maxNumFramesOnVideo, videoSegmentLength, positionSegment, shuffle, patch_size):

     train_names,train_labels,train_num_frames, train_bbox_files, test_names, test_labels, test_num_frames, test_bbox_files = datasetUtils.train_test_videos(train_videos_path, test_videos_path, path_dataset)
  
     combined = list(zip(train_names, train_num_frames, train_bbox_files))
  
     combined_train_names, combined_val_names, train_labels, val_labels = train_test_split(combined, train_labels, stratify=train_labels, test_size=0.30)
     train_names, train_num_frames, train_bbox_files = zip(*combined_train_names)
     val_names, val_num_frames, val_bbox_files = zip(*combined_val_names)
     train_labels = datasetUtils.labels_2_binary(train_labels)
     val_labels = datasetUtils.labels_2_binary(val_labels)
     test_labels = datasetUtils.labels_2_binary(test_labels)
     util.print_balance(train_labels, 'train')
     util.print_balance(val_labels, 'val')
     util.print_balance(test_labels, 'test')
     
     
     image_datasets = {
          "train": denseDataset.DenseDataset( dataset=train_names, labels=train_labels, numFrames=train_num_frames, bbox_files=train_bbox_files, spatial_transform=transforms_t["train"],
               nDynamicImages=numDiPerVideos, maxNumFramesOnVideo=maxNumFramesOnVideo, videoSegmentLength=videoSegmentLength, positionSegment=positionSegment, patch_size=patch_size),
          "val": denseDataset.DenseDataset( dataset=val_names, labels=val_labels, numFrames=val_num_frames, bbox_files=val_bbox_files, spatial_transform=transforms_t["val"], 
               nDynamicImages=numDiPerVideos, maxNumFramesOnVideo=maxNumFramesOnVideo, videoSegmentLength=videoSegmentLength, positionSegment=positionSegment, patch_size=patch_size),
          "test": denseDataset.DenseDataset( dataset=test_names, labels=test_labels, numFrames=test_num_frames, bbox_files=test_bbox_files, spatial_transform=transforms_t["test"],
               nDynamicImages=numDiPerVideos, maxNumFramesOnVideo=maxNumFramesOnVideo, videoSegmentLength=videoSegmentLength, positionSegment=positionSegment, patch_size=patch_size)
     }
     dataloaders_dict = {
          "train": torch.utils.data.DataLoader(image_datasets["train"], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers),
          "val": torch.utils.data.DataLoader( image_datasets["val"], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers),
          "test": torch.utils.data.DataLoader( image_datasets["test"], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
     }
     return dataloaders_dict, test_names
                    
def dataloader_test_dense_dataset(path_dataset, train_videos_path, test_videos_path, batch_size, num_workers, numDiPerVideos,
                         transforms_t, maxNumFramesOnVideo, videoSegmentLength, positionSegment, shuffle, patch_size):
     test_names, test_labels, test_num_frames, test_bbox_files = datasetUtils.only_anomaly_test_videos(test_videos_path, path_dataset)
     test_labels = datasetUtils.labels_2_binary(test_labels)
     util.print_balance(test_labels, 'test')
     image_datasets = {
          "test": denseDataset.DenseDataset( dataset=test_names, labels=test_labels, numFrames=test_num_frames, bbox_files=test_bbox_files, spatial_transform=transforms_t["test"],
               nDynamicImages=numDiPerVideos, maxNumFramesOnVideo=maxNumFramesOnVideo, videoSegmentLength=videoSegmentLength, positionSegment=positionSegment, patch_size=patch_size)
     }
     dataloaders_dict = {
          "test": torch.utils.data.DataLoader( image_datasets["test"], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
     }
     return dataloaders_dict, test_names


def initialize_final_only_test_anomaly_dataset(path_dataset, train_videos_path, test_videos_path, batch_size, num_workers,
                    numDiPerVideos, transforms_t, maxNumFramesOnVideo, videoSegmentLength, positionSegment, shuffle, getRawFrames, overlapping):
     # print('ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff')
     test_names, test_labels, test_num_frames, test_bbox_files = datasetUtils.only_anomaly_test_videos(test_videos_path, path_dataset)
     test_labels = datasetUtils.labels_2_binary(test_labels)
     util.print_balance(test_labels, 'test')

     
     dataset = anomalyDataset.AnomalyDataset( dataset=test_names, labels=test_labels, numFrames=test_num_frames, bbox_files=test_bbox_files, spatial_transform=transforms_t["test"],
               nDynamicImages=numDiPerVideos, maxNumFramesOnVideo=maxNumFramesOnVideo, videoSegmentLength=videoSegmentLength,
               positionSegment=positionSegment, getRawFrames=getRawFrames, overlapping=overlapping)
     
     image_datasets = {
          "test": dataset
     }
     dataloaders_dict = {
          "test": torch.utils.data.DataLoader( image_datasets["test"], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
     }
     return dataloaders_dict, test_names, dataset

def initialize_final_only_test_online_anomaly_dataset(path_dataset, test_videos_path, batch_size, num_workers, videoBlockLength,
                    numDynamicImgsPerBlock, transform, videoSegmentLength, shuffle,overlappingBlock, overlappingSegment):
     test_names, test_labels, test_num_frames, test_bbox_files = datasetUtils.only_anomaly_test_videos(test_videos_path, path_dataset)
     test_labels = datasetUtils.labels_2_binary(test_labels)
     util.print_balance(test_labels, 'test')

     # dataset, labels, numFrames, bbox_files, spatial_transform, videoBlockLength, numDynamicImgsPerBlock,
     #           videoSegmentLength, overlappingBlock, overlappingSegment
     dataset = anomalyOnlineDataset.AnomalyOnlineDataset(dataset=test_names, labels=test_labels, numFrames=test_num_frames, bbox_files=test_bbox_files,
                         spatial_transform=transform, videoBlockLength = videoBlockLength, numDynamicImgsPerBlock=numDynamicImgsPerBlock, videoSegmentLength=videoSegmentLength,
                         overlappingBlock=overlappingBlock, overlappingSegment=overlappingSegment)

     dataloader =  torch.utils.data.DataLoader( dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
     
     return dataloader, test_names, dataset

def initialize_test_anomaly_dataset(path_dataset, test_videos_path, batch_size, num_workers, videoBlockLength,
                    numDynamicImgsPerBlock, transform, videoSegmentLength, shuffle,overlappingBlock, overlappingSegment):
     test_names, test_labels, test_num_frames, test_bbox_files = datasetUtils.test_videos(test_videos_path, path_dataset)
     test_labels = datasetUtils.labels_2_binary(test_labels)
     util.print_balance(test_labels, 'test')

     # dataset, labels, numFrames, bbox_files, spatial_transform, videoBlockLength, numDynamicImgsPerBlock,
     #           videoSegmentLength, overlappingBlock, overlappingSegment
     dataset = anomalyOnlineDataset.AnomalyOnlineDataset(dataset=test_names, labels=test_labels, numFrames=test_num_frames, bbox_files=test_bbox_files,
                         spatial_transform=transform, videoBlockLength = videoBlockLength, numDynamicImgsPerBlock=numDynamicImgsPerBlock, videoSegmentLength=videoSegmentLength,
                         overlappingBlock=overlappingBlock, overlappingSegment=overlappingSegment)

     dataloader =  torch.utils.data.DataLoader( dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
     
     return dataloader, test_names, test_labels, dataset

def waqas_anomaly_downloader(videos,labels, numFrames, batch_size, num_workers, videoBlockLength,
                    numDynamicImgsPerBlock, transform, videoSegmentLength, shuffle,overlappingBlock, overlappingSegment, tmp_gtruth):
    
     dataset = anomalyOnlineDataset.AnomalyOnlineDataset(dataset=videos, labels=labels, numFrames=numFrames, bbox_files=[],
                         spatial_transform=transform, videoBlockLength = videoBlockLength, numDynamicImgsPerBlock=numDynamicImgsPerBlock, videoSegmentLength=videoSegmentLength,
                         overlappingBlock=overlappingBlock, overlappingSegment=overlappingSegment,temporal_gts=tmp_gtruth)

     dataloader =  torch.utils.data.DataLoader( dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
     
     return dataloader, dataset

def initialize_final_anomaly_dataset(path_dataset, train_videos_path, test_videos_path, batch_size, num_workers, numDiPerVideos,
                                        transforms_t, maxNumFramesOnVideo, videoSegmentLength, positionSegment, shuffle):
     train_names, train_labels, train_num_frames, train_bbox_files, test_names, test_labels, test_num_frames, test_bbox_files = datasetUtils.train_test_videos(train_videos_path, test_videos_path, path_dataset)
     combined = list(zip(train_names, train_labels, train_num_frames, train_bbox_files))
     random.shuffle(combined)

     train_names[:], train_labels[:], train_num_frames, train_bbox_files = zip(*combined)
     train_labels = datasetUtils.labels_2_binary(train_labels)
     test_labels = datasetUtils.labels_2_binary(test_labels)
     util.print_balance(train_labels, 'train')
     util.print_balance(test_labels, 'test')
     image_datasets = {
          "train": anomalyDataset.AnomalyDataset( dataset=train_names, labels=train_labels, numFrames=train_num_frames, bbox_files = train_bbox_files, spatial_transform=transforms_t["train"],
               nDynamicImages=numDiPerVideos, maxNumFramesOnVideo=maxNumFramesOnVideo, videoSegmentLength=videoSegmentLength, positionSegment=positionSegment),
          "test": anomalyDataset.AnomalyDataset( dataset=test_names, labels=test_labels, numFrames=test_num_frames,  bbox_files = test_bbox_files, spatial_transform=transforms_t["test"],
               nDynamicImages=numDiPerVideos, maxNumFramesOnVideo=maxNumFramesOnVideo, videoSegmentLength=videoSegmentLength, positionSegment=positionSegment)
     }
     dataloaders_dict = {
          "train": torch.utils.data.DataLoader(image_datasets["train"], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers),
          "test": torch.utils.data.DataLoader( image_datasets["test"], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
     }
     return dataloaders_dict, test_names

def initialize_train_val_anomaly_dataset(path_dataset, train_videos_path, test_videos_path, batch_size, num_workers,
                                        numDiPerVideos, transforms, maxNumFramesOnVideo, videoSegmentLength, positionSegment, shuffle):
     
     train_names,train_labels,train_num_frames, train_bbox_files, test_names, test_labels, test_num_frames, test_bbox_files = datasetUtils.train_test_videos(train_videos_path, test_videos_path, path_dataset)
  
     combined = list(zip(train_names, train_num_frames, train_bbox_files))
  
     combined_train_names, combined_val_names, train_labels, val_labels = train_test_split(combined, train_labels, stratify=train_labels, test_size=0.30)
     train_names, train_num_frames, train_bbox_files = zip(*combined_train_names)
     val_names, val_num_frames, val_bbox_files = zip(*combined_val_names)
     train_labels = datasetUtils.labels_2_binary(train_labels)
     val_labels = datasetUtils.labels_2_binary(val_labels)
     test_labels = datasetUtils.labels_2_binary(test_labels)
     util.print_balance(train_labels, 'train')
     util.print_balance(val_labels, 'val')
     util.print_balance(test_labels, 'test')
     # combined = list(zip(train_names, train_labels, train_num_frames))
     # random.shuffle(combined)
     # train_names[:], train_labels[:], train_num_frames = zip(*combined)
     #     combined = list(zip(test_names, test_labels, test_num_frames))
     #     random.shuffle(combined)
     #     test_names[:], test_labels[:], test_num_frames = zip(*combined)
     
     # print(len(train_names), len(train_labels), len(train_num_frames), len(test_names), len(test_labels), len(test_num_frames))
     
     
     image_datasets = {
          "train": anomalyDataset.AnomalyDataset( dataset=train_names, labels=train_labels, numFrames=train_num_frames, bbox_files=train_bbox_files, spatial_transform=transforms["train"],
               nDynamicImages=numDiPerVideos, maxNumFramesOnVideo=maxNumFramesOnVideo, videoSegmentLength=videoSegmentLength, positionSegment=positionSegment),
          "val": anomalyDataset.AnomalyDataset( dataset=val_names, labels=val_labels, numFrames=val_num_frames, bbox_files=val_bbox_files, spatial_transform=transforms["val"], 
               nDynamicImages=numDiPerVideos, maxNumFramesOnVideo=maxNumFramesOnVideo, videoSegmentLength=videoSegmentLength, positionSegment=positionSegment),
          "test": anomalyDataset.AnomalyDataset(dataset=test_names, labels=test_labels, numFrames=test_num_frames, bbox_files=test_bbox_files,
               spatial_transform=transforms["test"], nDynamicImages=numDiPerVideos, maxNumFramesOnVideo=maxNumFramesOnVideo,
               videoSegmentLength=videoSegmentLength, positionSegment=positionSegment)
     }
     dataloaders_dict = {
          "train": torch.utils.data.DataLoader(image_datasets["train"], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers),
          "val": torch.utils.data.DataLoader( image_datasets["val"], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers),
          "test": torch.utils.data.DataLoader( image_datasets["test"], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
     }
     return dataloaders_dict, test_names


def initialize_train_aumented_anomaly_dataset(path_dataset, batch_size, num_workers, transforms, shuffle, val_split):
     
     train_names, train_labels = datasetUtils.train_test_videos_aumented(path_dataset)
     # combined = list(zip(train_names, train_num_frames))
     train_names, val_names, train_labels, val_labels = train_test_split(train_names, train_labels, stratify=train_labels, test_size=val_split)

     # print(len(train_names), len(train_labels))
     # print(len(train_names), len(train_labels))
     util.print_balance(train_labels, 'train')
     util.print_balance(val_labels, 'val')

     image_datasets = {
          "train": anomalyDatasetAumented.AnomalyDatasetAumented(images=train_names, labels=train_labels, spatial_transform=transforms["train"]),
          "val": anomalyDatasetAumented.AnomalyDatasetAumented(images=val_names, labels=val_labels, spatial_transform=transforms["val"])
     }
     dataloaders_dict = {
          "train": torch.utils.data.DataLoader(image_datasets["train"], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers),
          "val": torch.utils.data.DataLoader(image_datasets["val"], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
     }
     return dataloaders_dict