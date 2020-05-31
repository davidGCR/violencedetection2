class ResultPolicy():
    def __init__(self):
        self.lowest_train_loss = 0
        self.train_acc = 0
        self.lowest_test_loss = 0
        self.best_test_acc = 0
        self.bestEpoch = 0
    
    def update(self, train_loss, train_acc, test_loss, test_acc, epoch):
        if test_loss <= train_loss and test_acc > self.best_test_acc:
            self.best_test_acc = test_acc
            self.lowest_test_loss = test_loss
            self.lowest_train_loss = train_loss
            self.best_epoch = epoch
    
    def getFinalTestAcc(self):
        return self.best_test_acc
    
    def getFinalTestLoss(self):
        return self.lowest_test_loss
    
    def getFinalEpoch(self):
        return self.best_epoch


