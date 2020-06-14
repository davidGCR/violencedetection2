
from VIOLENCE_DETECTION.violenceDataset import ViolenceDataset
import torch

class Dataloader():
    def __init__(self, X, y, numFrames, transform,  **kargs):
        self.X = X
        self.y = y
        self.numFrames = numFrames
        self.transform = transform
        self.kargs = kargs
        print(self.kargs)
        # self.dataset, self.dataloader = self.buildDataloader()
        self.dataset = ViolenceDataset(dataset=self.X,
                                labels=self.y,
                                numFrames=self.numFrames,
                                spatial_transform=self.transform,
                                numDynamicImagesPerVideo=self.kargs['NDI'],
                                videoSegmentLength=self.kargs['videoSegmentLength'],
                                positionSegment=self.kargs['positionSegment'],
                                overlaping=self.kargs['overlapping'],
                                frame_skip=self.kargs['frameSkip'],
                                skipInitialFrames=self.kargs['skipInitialFrames'], ppType=self.kargs['pptype'])
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.kargs['batchSize'], shuffle=self.kargs['shuffle'], num_workers=self.kargs['numWorkers'])
    
    def buildDataloader(self):
        dataset = ViolenceDataset(dataset=self.X,
                                labels=self.y,
                                numFrames=self.numFrames,
                                spatial_transform=self.transform,
                                numDynamicImagesPerVideo=self.kargs['NDI'],
                                videoSegmentLength=self.kargs['videoSegmentLength'],
                                positionSegment=self.kargs['positionSegment'],
                                overlaping=self.kargs['overlapping'],
                                frame_skip=self.kargs['frameSkip'],
                                skipInitialFrames=self.kargs['skipInitialFrames'],
                                pptype=self.kargs['pptype'])

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.kargs['batchSize'], shuffle=self.kargs['shuffle'], num_workers=self.kargs['numWorkers'])
        return dataset, dataloader

    def getDataloader(self):
        return self.dataloader
    
    def getElement(self, name, label, savePath, ndi, seqLen, transform, ptype):
        index, full_name = self.dataset.getindex(vid_name=name, label=label)
        frames, dimages, dimages2, label = None, None, None, None
        if index is not None:
            frames, dimages, dimages2, label = self.dataset.getOneItem(index, transform=transform, ptype=ptype, savePath=savePath, ndi=ndi, seqLen=seqLen)

        return frames, dimages, dimages2, label


