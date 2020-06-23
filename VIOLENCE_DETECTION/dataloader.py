
from VIOLENCE_DETECTION.violenceDataset import ViolenceDataset
import torch

class MyDataloader():
    def __init__(self, kargs):
        self.kargs = kargs
        self.X = self.kargs['X']
        self.y = self.kargs['y']
        self.numFrames = self.kargs['numFrames']
        self._transform = self.kargs['transform']
        # print(self.kargs)
        # self.dataset, self.dataloader = self.buildDataloader()
        self._dataset = ViolenceDataset(dataset=self.X,
                                labels=self.y,
                                numFrames=self.numFrames,
                                spatial_transform=self.transform,
                                numDynamicImagesPerVideo=self.kargs['NDI'],
                                videoSegmentLength=self.kargs['videoSegmentLength'],
                                positionSegment=self.kargs['positionSegment'],
                                overlaping=self.kargs['overlapping'],
                                frame_skip=self.kargs['frameSkip'],
                                skipInitialFrames=self.kargs['skipInitialFrames'],
                                ppType=self.kargs['pptype'])
        self._dataloader = None
    
    @property
    def dataset(self): 
        return self._dataset
    @dataset.setter
    def dataset(self, x):
        self._dataset = x

    @property
    def dataloader(self):
        if self._dataloader is None:
            self._dataloader = torch.utils.data.DataLoader(self._dataset, batch_size=self.kargs['batchSize'], shuffle=self.kargs['shuffle'], num_workers=self.kargs['numWorkers'])
        return self._dataloader
    @dataloader.setter
    def dataloader(self, x):
        self._dataloader = x

    @property
    def transform(self): 
        return self._transform
    @transform.setter
    def transform(self, x):
        self._dataloader = None
        self._transform = x
        self._dataset.setTransform(self._transform)
    
    # def getElement(self, name, label, savePath, ndi, seqLen, transform, ptype):
    #     index, full_name = self.dataset.getindex(vid_name=name, label=label)
    #     frames, dimages, dimages2, label = None, None, None, None
    #     if index is not None:
    #         frames, dimages, dimages2, label = self.dataset.getOneItem(index, transform=transform, ptype=ptype, savePath=savePath, ndi=ndi, seqLen=seqLen)

    #     return frames, dimages, dimages2, label


