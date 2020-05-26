import torch
import torchvision
# from tensorboardcolab import TensorBoardColab
import time
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self, model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, device, num_epochs, checkpoint_path,
                    numDynamicImage, plot_samples, train_type, save_model):
        self.model = model
        # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
        # self.model_name = "alexnet"
        # Number of classes in the dataset
        self.num_classes = 2
        self.train_type = train_type
        # Flag for feature extracting. When False, we finetune the whole model,
        #   when True we only update the reshaped layer params
        self.feature_extract = True

        self.input_size = 224
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.criterion = criterion
        # self.tb = TensorBoardColab()
        self.device = device
        self.num_epochs = num_epochs
        self.scheduler = scheduler
        self._checkpoint_path = checkpoint_path
        # self.best_model_wts = copy.deepcopy(model.state_dict())
        self.best_acc = 0.0
        self.numDynamicImages = numDynamicImage
        self.plot_samples = plot_samples
        self.train_best_acc = 0
        self._save_model = save_model
        # self.writer = SummaryWriter('runs/anomaly')
        # self.minibatch = 0

    @property
    def save_model(self):
        return self._save_model
    @save_model.setter
    def save_model(self, value):
        self._save_model = value
    
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
                
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
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
        # if self.train_type == constants.OPERATION_TRAINING_FINAL:
        
        
        # self.tb.save_value("trainLoss", "train_loss", epoch, epoch_loss)
        # self.tb.save_value("trainAcc", "train_acc", epoch, epoch_acc)

        # running_loss += loss.item()
        # i = 
        
        return epoch_loss, epoch_acc

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
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
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
        if epoch_acc > self.train_best_acc:
            self.train_best_acc = epoch_acc.item()
            # best_model_wts = copy.deepcopy(self.model.state_dict())

            # checkpoint_name = self.checkpoint_path+'.pth'
            # self._checkpoint_path = self._checkpoint_path+'-epoch-'+str(epoch)+'.tar'
            print('Saving model...',self._checkpoint_path+'-epoch-'+str(epoch)+'.tar')
            torch.save(self.model, self._checkpoint_path+'-epoch-'+str(epoch)+'.tar')    

        return epoch_loss, epoch_acc
