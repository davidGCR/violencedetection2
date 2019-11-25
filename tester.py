import time
import copy
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

class Tester:
    def __init__(self, model, dataloader, device, numDiPerVideos, plot_samples=False):
        self.model = model  
        self.dataloader = dataloader
        # self.optimizer = optimizer
        self.plot_samples = plot_samples
        self.numDiPerVideos = numDiPerVideos
        # self.tb = TensorBoardColab()
        self.device = device

    def test_model(self):
        self.model.eval()
        gt_labels = []
        predictions = []  #indeicea
        scores = []
        # Iterate over data.
        for inputs, labels, video_names in tqdm(self.dataloader):
            if self.numDiPerVideos > 1:
                inputs = inputs.permute(1, 0, 2, 3, 4)
            gt_labels.extend(labels.numpy())
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            # self.optimizer.zero_grad()
            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                p = torch.nn.functional.softmax(outputs, dim=1)
               
                # print('-- outputs size: ', outputs.size())
                # print('-- labels size: ',labels.size())
                # loss = self.criterion(outputs, labels)
                values, indices = torch.max(outputs, 1)
                scores.extend(p.cpu().numpy())
                predictions.extend(indices.cpu().numpy())
                # print('predictions: ',preds)
                # p2 = p[:,indices.cpu().numpy()]
                # print(indices,p,p2)

        return gt_labels, predictions, scores