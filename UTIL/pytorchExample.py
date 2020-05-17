import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import os
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import time

num_epochs = 30
board_folder = 'PytorchExample4'
writer = SummaryWriter('runs/' + board_folder)

def my_trainer(model, dataloaders, criterion, optimizer, exp_lr_scheduler, num_epochs):
    from trainer import Trainer
    # dataloaders_dict = {'train':trainloader, 'val': testloader}
    trainer = Trainer(model=model,
                    dataloaders=dataloaders,
                    criterion=criterion,
                    optimizer=optimizer,
                    scheduler=exp_lr_scheduler,
                    device=device,
                    num_epochs=num_epochs,
                    checkpoint_path=None,
                    numDynamicImage=1,
                    plot_samples=False,
                    train_type=None,
                    save_model=False)
    train_lost = []
    train_acc = []
    val_lost = []
    val_acc = []

    for epoch in range(1, num_epochs + 1):
        print("----- Epoch {}/{}".format(epoch, num_epochs))

        epoch_loss_train, epoch_acc_train = trainer.train_epoch(epoch)
        train_lost.append(epoch_loss_train)
        train_acc.append(epoch_acc_train)

        epoch_loss_val, epoch_acc_val = trainer.val_epoch(epoch)
        exp_lr_scheduler.step()
        val_lost.append(epoch_loss_val)
        val_acc.append(epoch_acc_val)
        
        writer.add_scalar('training loss', epoch_loss_train, epoch)
        writer.add_scalar('validation loss', epoch_loss_val, epoch)

        writer.add_scalar('training Acc', epoch_acc_train, epoch)
        writer.add_scalar('validation Acc', epoch_acc_val, epoch)

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    # best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                # best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'train':
                writer.add_scalar('training loss', epoch_loss, epoch+1)
                writer.add_scalar('training Acc', epoch_acc, epoch+1)
            elif phase == 'val':
                writer.add_scalar('validation loss', epoch_loss, epoch+1)
                writer.add_scalar('validation Acc', epoch_acc, epoch+1)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    # model.load_state_dict(best_model_wts)
    return model


# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'hymenoptera_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

##### model 1
# model_ft = models.resnet18(pretrained=True)
# num_ftrs = model_ft.fc.in_features
# model_ft.fc = nn.Linear(num_ftrs, 2)
# model_ft = model_ft.to(device)

##### model 2
model_conv = models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False
# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)
model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

num_epochs=30
model_ft = train_model(model_conv, dataloaders, dataset_sizes, criterion, optimizer, exp_lr_scheduler,
                       num_epochs=num_epochs)
