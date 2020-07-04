import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_acc = np.Inf
        self.best_epoch = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, val_acc, epoch, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            # self.best_loss = val_loss
            self.best_epoch = epoch
            self.save_checkpoint(val_loss, val_acc, epoch, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # self.best_loss = val_loss
            # self.best_epoch = epoch
            self.save_checkpoint(val_loss, val_acc, epoch, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_acc, epoch, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}), Validation accuracy ({self.best_acc:.6f} --> {val_acc:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), self.path)
        torch.save({
            'epoch': epoch,
            'val_acc': val_acc,
            'val_loss': val_loss,
            'model_state_dict': model.state_dict()
            }, self.path)
        self.val_loss_min = val_loss
        self.best_acc = val_acc
        self.best_epoch = epoch
