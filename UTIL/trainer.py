import torch
import torchvision
# from tensorboardcolab import TensorBoardColab
import time
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
from constants import DEVICE

class Trainer:
    def __init__(self, model, model_transfer, train_dataloader, val_dataloader, criterion, optimizer, num_epochs, checkpoint_path):
        self.model = model
        if model_transfer is not None:
            if DEVICE == 'cuda:0':
                self.model.load_state_dict(torch.load(model_transfer), strict=False)
            else:
                self.model.load_state_dict(torch.load(model_transfer, map_location=DEVICE))
        
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.criterion = criterion

        self.num_epochs = num_epochs
        self._checkpoint_path = checkpoint_path

    
    @property
    def checkpoint_path(self):
        return self._checkpoint_path
    @checkpoint_path.setter
    def checkpoint_path(self, value):
        self._checkpoint_path = value
    
    # @property
    # def dataloaders(self):
    #     return self._dataloaders
    # @dataloaders.setter
    # def dataloaders(self, value):
    #     self._dataloaders = value

    def getModel(self):
        return self.model

    def train_epoch(self, epoch):
        # self.scheduler.step(epoch)
        self.model.train()  # Set model to training mode
        # is_inception = False
        running_loss = 0.0
        running_corrects = 0
        padding = 5
        
        # Iterate over data.
        # for i, data in enumerate(tqdm(self.dataloaders["train"])):
        for i, data in enumerate(tqdm(self.train_dataloader)): #inputs, labels:  <class 'torch.Tensor'> torch.Size([64, 3, 224, 224]) <class 'torch.Tensor'> torch.Size([64])
            # print('inputs, labels: ',type(inputs),inputs.size(), type(labels), labels.size())
            # print('inputs trainer: ', inputs.size())
            # print(video_names, labels)
            
            inputs, labels, _, _ = data
            # print(inputs.size())
            batch_size = inputs.size()[0]
            # if self.numDynamicImages > 1:
            #     inputs = inputs.permute(1, 0, 2, 3, 4)
                
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            # zero the parameter gradients
            self.optimizer.zero_grad()
            # track history if only in train
            with torch.set_grad_enabled(True):    
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                loss.backward()
                self.optimizer.step()

            # print('Train Lost: ', outputs, labels, loss.item())
            running_loss += loss.item() * batch_size
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(self.train_dataloader.dataset)
        epoch_acc = running_corrects.double() / len(self.train_dataloader.dataset)

        print("{} Loss: {:.4f} Acc: {:.4f}".format('train', epoch_loss, epoch_acc))
        
        return epoch_loss, epoch_acc.item()

    def val_epoch(self, epoch):
        running_loss = 0.0
        running_corrects = 0
        self.model.eval()
        
        # Iterate over data.
        for inputs, labels, video_names, _ in self.val_dataloader:
        # for inputs, labels  in self.dataloaders["val"]:
            batch_size = inputs.size()[0]
            # if self.numDynamicImages > 1:
            #     inputs = inputs.permute(1, 0, 2, 3, 4)
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            # zero the parameter gradients
            # self.optimizer.zero_grad()
            # forward
            # track history if only in train
            # with torch.set_grad_enabled(False):
            with torch.no_grad():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                # print('Validation Lost: ', outputs, labels, loss.item())

                _, preds = torch.max(outputs, 1)

                # statistics
                running_loss += loss.item() * batch_size
                # running_loss += loss.item() 
                running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(self.val_dataloader.dataset)
        epoch_acc = running_corrects.double() / len(self.val_dataloader.dataset)

        print("{} Loss: {:.4f} Acc: {:.4f}".format("val", epoch_loss, epoch_acc))
        # if epoch_acc > self.train_best_acc:
        #     self.train_best_acc = epoch_acc.item()
            # best_model_wts = copy.deepcopy(self.model.state_dict())

            # checkpoint_name = self.checkpoint_path+'.pth'
            # self._checkpoint_path = self._checkpoint_path+'-epoch-'+str(epoch)+'.tar'
            

        return epoch_loss, epoch_acc.item()
    
    def saveCheckpoint(self, epoch, flac):
        if flac:
            print('Saving model...',self._checkpoint_path+'-epoch-'+str(epoch)+'.pth')
            # torch.save(self.model, self._checkpoint_path+'-epoch-'+str(epoch)+'.tar')    
            torch.save(self.model.state_dict(), self._checkpoint_path+'-epoch-'+str(epoch)+'.pth')

