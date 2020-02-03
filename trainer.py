import torch
import torchvision
# from tensorboardcolab import TensorBoardColab
import time
import copy
from util import save_checkpoint, imshow
from tqdm import tqdm
import dynamicImage
import matplotlib.pyplot as plt
import constants


class Trainer:
    def __init__(self, model, dataloaders, criterion, optimizer, scheduler, device, num_epochs, checkpoint_path, numDynamicImage, plot_samples, train_type):
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
        self.dataloaders = dataloaders
        self.optimizer = optimizer
        self.criterion = criterion
        # self.tb = TensorBoardColab()
        self.device = device
        self.num_epochs = num_epochs
        self.scheduler = scheduler
        self.checkpoint_path = checkpoint_path
        self.best_model_wts = copy.deepcopy(model.state_dict())
        self.best_acc = 0.0
        self.numDynamicImages = numDynamicImage
        self.plot_samples = plot_samples
        self.train_best_acc = 0

    def train_epoch(self, epoch):
        # self.scheduler.step(epoch)
        self.model.train()  # Set model to training mode
        # is_inception = False
        running_loss = 0.0
        running_corrects = 0
        padding = 5
        
        # Iterate over data.
        for inputs, labels, video_names, bbox_segments in tqdm(self.dataloaders["train"]): #inputs, labels:  <class 'torch.Tensor'> torch.Size([64, 3, 224, 224]) <class 'torch.Tensor'> torch.Size([64])
            # print('inputs, labels: ',type(inputs),inputs.size(), type(labels), labels.size())
            # print('inputs trainer: ', inputs.size())
            print(video_names, labels)
            if self.plot_samples:
                print(video_names)
                plt.figure(figsize=(10,12))
                images = torchvision.utils.make_grid(inputs.cpu().data, padding=padding)
                imshow(images, video_names)
                dyImg = dynamicImage.computeDynamicImages(str(video_names[0]), self.numDynamicImages,16)
                dis = torchvision.utils.make_grid(dyImg.cpu().data, padding=padding)
                # print(video_names[0])
                plt.figure(figsize=(10,12))
                imshow(dis, 'RAW - '+str(video_names[0]))
            if self.numDynamicImages > 1:
                # print('==== dataloader size: ',inputs.size()) #[batch, ndi, ch, h, w]
                inputs = inputs.permute(1, 0, 2, 3, 4)
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            # zero the parameter gradients
            self.optimizer.zero_grad()
            # track history if only in train
            with torch.set_grad_enabled(True):    
                outputs = self.model(inputs)
                # print('-- outputs size: ', outputs.size(), outputs)
                # print('-- labels size: ',labels.size(), labels)
                loss = self.criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                # backward + optimize only if in training phase
                loss.backward()
                self.optimizer.step()
                # self.scheduler.step(epoch)
            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(self.dataloaders["train"].dataset)
        epoch_acc = running_corrects.double() / len(self.dataloaders["train"].dataset)

        print("{} Loss: {:.4f} Acc: {:.4f}".format('train', epoch_loss, epoch_acc))
        if self.train_type == constants.OPERATION_TRAINING_FINAL:
            if epoch_acc > self.train_best_acc:
                self.train_best_acc = epoch_acc.item()
                checkpoint_name = self.checkpoint_path+'.pth'
                print('Saving FINAL model...',checkpoint_name)
                torch.save(self.model, checkpoint_name)
        # self.tb.save_value("trainLoss", "train_loss", epoch, epoch_loss)
        # self.tb.save_value("trainAcc", "train_acc", epoch, epoch_acc)
        return epoch_loss, epoch_acc

    def val_epoch(self, epoch):
        running_loss = 0.0
        running_corrects = 0
        self.model.eval()
        
        # Iterate over data.
        for inputs, labels, video_names, bbox_segments in self.dataloaders["val"]:
            if self.numDynamicImages > 1:
                inputs = inputs.permute(1, 0, 2, 3, 4)
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            # zero the parameter gradients
            self.optimizer.zero_grad()
            # forward
            # track history if only in train
            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                # print('-- outputs size: ', outputs.size())
                # print('-- labels size: ',labels.size())
                loss = self.criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(self.dataloaders["val"].dataset)
        epoch_acc = running_corrects.double() / len(self.dataloaders["val"].dataset)

        print("{} Loss: {:.4f} Acc: {:.4f}".format("val", epoch_loss, epoch_acc))
        if self.checkpoint_path != None and epoch_acc > self.best_acc:
            self.best_acc = epoch_acc
            checkpoint_name = self.checkpoint_path+'.pth'
            print('Saving entire model...',checkpoint_name)
            torch.save(self.model, checkpoint_name)
            # torch.save(self.model.state_dict(),self.checkpoint_path)
            # save_checkpoint(self.model, self.checkpoint_path)
        # self.tb.save_value("testLoss", "test_loss", epoch, epoch_loss)
        # self.tb.save_value("testAcc", "test_acc", epoch, epoch_acc)

        return epoch_loss, epoch_acc
