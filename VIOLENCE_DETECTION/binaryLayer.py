import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
import scipy.io as sio
import constants

import matplotlib.pyplot as plt
from sklearn import svm, metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV

def _load_data(path):
    mat_data = sio.loadmat(path)
    R = mat_data['R']
    XXrot = mat_data['XXrot']
    Y = mat_data['Y']#36000x8
    Ycompact = mat_data['Ycompact']
    labels = mat_data['labels'][0]

    # print(R, type(R), R.shape)
    # print(type(XXrot), XXrot.shape, XXrot[0,:])
    # print('Y=', type(Y), Y.shape, Y[0,:], type(Y[0,0]))
    # print(R, type(R), R.shape)
    # R = torch.from_numpy(train_data).float()
    return Y, labels

def patterns2Numbers(data, h, w):
    samples_binary = []
    samples_decimal = []
    l = h * w
    for i in range(0, data.shape[0], l):
        f_map = data[i:i+l,:] #36x8
        samples_binary.append(f_map)
        f_map_dec = []
        for j in range(l):
            dec = binaryToDecimal(f_map[j,:])
            f_map_dec.append(dec)
        samples_decimal.append(f_map_dec)
        
    return samples_binary, samples_decimal

def binaryToDecimalN(binary):
    binary1 = binary 
    decimal, i, n = 0, 0, 0
    while(binary != 0): 
        dec = binary % 10
        decimal = decimal + dec * pow(2, i) 
        binary = binary//10
        i += 1
    print(decimal)
    return decimal

def binaryToDecimal(pattern):
    decimal = 0
    for i, bit in enumerate(pattern):
        decimal += bit*pow(2, len(pattern)-i-1)
    # print(decimal)
    return decimal

def main():
    # data_binary, labels = _load_data(os.path.join(constants.PATH_RESULTS, 'HOCKEY', 'ITQdata', 'ITQfeatures_out_iter=10.mat'))
    path = os.path.join('/Users/davidchoqueluqueroman/Google Drive/ITQData','ITQfeatures_ft=No_out_bits=7_iter=50.mat')
    data_binary, labels = _load_data(path)
    (nsamples, bits) = data_binary.shape
    print('File: ',path,' Loaded data ',data_binary.shape,'labels ', len(labels))
    data_binary, data_decimal = patterns2Numbers(data_binary, h=6, w=6)
    # train_bits= data_binary[0:800]
    # test_bits = data_binary[800: len(data_binary)]
    # train_dec= data_decimal[0:800]
    # test_dec = data_decimal[800:len(data_decimal)]

    # samples = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # for i in samples:
    #     print('data_binary ', len(data_binary), 'data_binary[i] ', type(data_binary[i]), data_binary[i].shape)
    #     print('data_decimal ', len(data_decimal), 'data_decimal[i] ', type(data_decimal[i]), len(data_decimal[i]))
    #     print('label ', labels[i])
    #     sample = data_decimal[i]
    #     print(sample)
        
    #     (hist, _) = np.histogram(sample, bins=256, range=(0, 255))
        
    #     # An "interface" to matplotlib.axes.Axes.hist() method
    #     n, bins, patches = plt.hist(x=sample, bins=256, range=(0, 255), color='#0504aa', alpha=0.7, rwidth=0.85)
    #     print('histogram=', n)
    #     print('histogram numpy=', hist)
    #     # print('bins=', bins)
        
    #     plt.grid(axis='y', alpha=0.75)
    #     plt.xlabel('Value')
    #     plt.ylabel('Frequency')
    #     plt.title('Histogram-label={}'.format(labels[i]))
    #     plt.text(23, 45, r'$\mu=15, b=3$')
    #     maxfreq = n.max()
    #     # Set a clean upper y-axis limit.
    #     plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    #     plt.show()
    ##Histograms
    X_hist = []
    
    for dec in data_decimal:
        (hist, _) = np.histogram(dec, bins=2**bits, range=(0,2**bits -1))
        X_hist.append(hist)
    print('Histogram bins=', len(X_hist[3]))
    X_train = X_hist[0:800]
    X_test = X_hist[800: len(X_hist)]
    y_train= labels[0:800]
    y_test = labels[800:len(labels)]

    param_grid = {'C': [0.1, 1, 10, 100, 1000],  
                'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
                'kernel': ['rbf']}
    grid = GridSearchCV(svm.SVC(), param_grid, refit = True, verbose = 3) 
    # fitting the model for grid search 
    grid.fit(X_train, y_train)
    # print best parameter after tuning 
    print(grid.best_params_)
    grid_predictions = grid.predict(X_test) 
    # print classification report 
    print(classification_report(y_test, grid_predictions)) 
    
    # kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    kernels = ['rbf']
    for kernel in kernels:
        print('****** ',kernel)
        #Create a svm Classifier
        # clf = svm.SVC(kernel=kernel)  # Linear Kernel
        clf = svm.SVC(C=10, gamma=0.0001, kernel=kernel)
        scores = cross_val_score(clf, X_hist, labels, cv=5)
        print(scores)
        #Train the model using the training sets
        # clf.fit(X_train, y_train)
        #Predict the response for test dataset
        # y_pred = clf.predict(X_test)
        # Model Accuracy: how often is the classifier correct?
        # print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    

if __name__ == "__main__":
    main()