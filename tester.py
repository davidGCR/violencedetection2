import time
import copy
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import numpy as np
from FPS import FPSMeter

class Tester:
    def __init__(self, model, dataloader, loss, device, numDiPerVideos, plot_samples=False):
        self.model = model  
        self.dataloader = dataloader
        # self.optimizer = optimizer
        self.plot_samples = plot_samples
        self.numDiPerVideos = numDiPerVideos
        # self.tb = TensorBoardColab()
        self.device = device
        self.loss = loss
        self.fpsMeter = FPSMeter()
        # self.model  = self.model.to(device)

    def test_model(self):
        self.model.eval()
        gt_labels = []
        predictions = []  #indeicea
        scores = []
        test_error = 0.0
        # Iterate over data.
        for inputs, labels, video_names, preprocessing_time in tqdm(self.dataloader):
            if self.numDiPerVideos > 1:
                inputs = inputs.permute(1, 0, 2, 3, 4)
            # gt_labels.extend(labels.numpy())
            gt_labels.extend(labels.numpy())
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            # self.optimizer.zero_grad()
            start_time = time.time()
            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                err = self.loss(outputs, labels)
                # print('err: ',err.item())
                test_error += err.item()*inputs.size(0)
                p = torch.nn.functional.softmax(outputs, dim=1)
                values, indices_preds = torch.max(outputs, 1)
                scores.extend(p.cpu().numpy())
                # predictions.extend(indices_preds.cpu().numpy())
                predictions.extend(indices_preds.cpu().numpy())
                torch.cuda.synchronize()
            end_time = time.time()
            inf_time = end_time - start_time
            self.fpsMeter.update(inf_time+preprocessing_time)

        test_error = test_error/len(self.dataloader.dataset)
        predictions = np.array(predictions)
        gt_labels = np.array(gt_labels)
        # test_errors = np.array(test_errors)
        self.fpsMeter.print_statistics()
        return predictions, scores, gt_labels, test_error, self.fpsMeter.fps()
        # return gt_labels, predictions, scores
    
    def predict(self, dynamic_img):
        self.model.eval()
        dynamic_img = dynamic_img.to(self.device)

        # print('dynamic_img: ', type(dynamic_img), dynamic_img.is_cuda, self.device)

        with torch.set_grad_enabled(False):
            # torch.cuda.synchronize()
            outputs = self.model(dynamic_img)
            
            p = torch.nn.functional.softmax(outputs, dim=1)
            p = p.cpu().numpy()
            values, indices = torch.max(outputs, 1)
            # print('Prediction: ', indices, type(indices))
        return indices.cpu().item(), p[0][1]

    def predict_time(self, dynamic_img, num_iter):
        self.model.eval()
        dynamic_img = dynamic_img.to(self.device)
        inference_times = []

        for i in range(num_iter):
            start_time = time.time()
            with torch.set_grad_enabled(False):
                # torch.cuda.synchronize()
                outputs = self.model(dynamic_img)
                
                p = torch.nn.functional.softmax(outputs, dim=1)
                p = p.cpu().numpy()
                values, indices = torch.max(outputs, 1)
                torch.cuda.synchronize()
            end_time = time.time()
            inference_times.append(end_time - start_time)
        inference_time = sum(inference_times) / len(inference_times)
        fps = 1.0 / inference_time
        print('FPS: ', fps, dynamic_img.size())
            # print('Prediction: ', indices, type(indices))
        return indices.cpu().item(), p[0][1], fps