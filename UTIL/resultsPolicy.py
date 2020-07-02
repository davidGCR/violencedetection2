class ResultPolicy():
    def __init__(self, patience, max_loss_difference):
        super().__init__()
        self.lowest_train_loss = 0.0
        self.train_acc = 0.0
        self.lowest_test_loss = 0.0
        self.best_test_acc = 0.0
        self.bestEpoch = 0
        self.difference = 0
        self.max_loss_difference = max_loss_difference
        self.patience = patience
        self.epoch_counter = 0
    
    def update(self, train_loss, train_acc, test_loss, test_acc, epoch):
        flac = test_loss <= train_loss and test_acc >= self.best_test_acc
        # stop = False
        if flac:
            self.best_test_acc = test_acc
            self.lowest_test_loss = test_loss
            self.lowest_train_loss = train_loss
            self.bestEpoch = epoch
        elif (test_loss - train_loss) > self.max_loss_difference:
            self.epoch_counter += 1
            
        stop = self.epoch_counter > self.patience 
        return flac, stop
    
    def getFinalTestAcc(self):
        return self.best_test_acc
    
    def getFinalTestLoss(self):
        return self.lowest_test_loss
    
    def getFinalEpoch(self):
        return self.bestEpoch


