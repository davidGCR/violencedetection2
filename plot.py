import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch 
import pandas as pd
import os
import argparse
from util import *


def fromCSV(file):
  df = pd.read_csv(file)
  llist = df['Value'].tolist()
  return llist
  
def loadList(name):
  with open(name, 'rb') as filehandle:
    # read the data as binary data stream
    hist2 = pickle.load(filehandle)
    return hist2
  
def getAverageFromFolds(llist, nepochs):
  arr = np.array(llist)
  arr = np.reshape(arr,(5,nepochs))
  arr = np.mean(arr, axis=0)
  return arr.tolist()
  
  
# print(len(train_lost))
def plotScalarFolds(listF,listL, tepochs,typ, fig2, rows,cols, num):
  for i in range(0,len(listF),tepochs):
  
    a = fig2.add_subplot(rows, cols, num)
    x_train_acc = np.array(listF[i:i+tepochs])
    
    plt.plot(np.arange(0, tepochs, 1),x_train_acc)
    plt.xlabel('Epoca')
    plt.ylabel('Tasa de Acierto')
    a.set_title(str(typ)+' - Tasa de Acierto')
    
    a = fig2.add_subplot(rows, cols, num+1)
    x_train_lost = np.array(listL[i:i+tepochs])
    plt.plot(np.arange(0, tepochs, 1),x_train_lost)
    plt.xlabel('Epoca')
    plt.ylabel('Error')
    a.set_title(str(typ)+' - Error')
    plt.legend(['Iteracion 1', 'Iteracion 2', 'Iteracion 3', 'Iteracion 4', 'Iteracion 5'], loc='upper right',fontsize='medium')

  
def plotScalarCombined(trainlist,testlist, tepochs,title, ylabel, fig2, rows, cols, num, lastEpoch):
  # # # # # # fig2 = plt.figure(figsize=(12,5))
  a= fig2.add_subplot(rows, cols, num)

  x = np.arange(0, tepochs, 1)
  plt.plot(x, trainlist, 'r')
  plt.plot(x, testlist, 'b')
  plt.xlabel('Epoca')
  plt.ylabel(ylabel)
  a.set_title(title)
  plt.legend(['Train', 'Test'], loc='upper right', fontsize='large')
  plt.axvline(x=lastEpoch, color='g', linestyle='--')
  


###From Pickle
# modelType = 'alexnetv2-frames-Finetuned:False-2-decay-'
# path = '/media/david/datos/Violence DATA/violentflows/Results/frames/'
# modelType = 'alexnetv2-frames-Finetuned:False-5-decay-'
# path = '/media/david/datos/Violence DATA/HockeyFights/Results/frames/'

def plot_results(path, lastEpoch, nfolds):
    train_lost = loadList(str(path)+'-train_lost.txt')
    train_acc = loadList(str(path)+'-train_acc.txt')
    test_lost = loadList(str(path)+'-val_lost.txt')
    test_acc = loadList(str(path)+'-val_acc.txt')

    num_epochs = int(len(train_lost)/nfolds)
    # num_epochs = 30
    print('len: ',len(train_lost))
    print('num_epochs size: ', num_epochs)

    # saveList(path+modelType+'train_lost.txt',train_lost[0:150])
    # saveList(path+modelType+'train_acc.txt',train_lost[150:300])
    # saveList(path+modelType+'test_lost.txt',train_lost[150:300])
    # saveList(path+modelType+'test_acc.txt',train_lost[150:300])

    fig2 = plt.figure(figsize=(12,12))

    plotScalarFolds(train_acc,train_lost,num_epochs,'Train',fig2,3,2,1)
    plotScalarFolds(test_acc, test_lost, num_epochs, 'Test',fig2,3,2,3)
    acc = 0
    if nfolds == 5:
      avgTrainAcc = getAverageFromFolds(train_acc,num_epochs)
      avgTrainLost = getAverageFromFolds(train_lost,num_epochs)
      avgTestAcc = getAverageFromFolds(test_acc,num_epochs)
      avgTestLost = getAverageFromFolds(test_lost, num_epochs)
      acc = np.max(avgTestAcc[0:lastEpoch+1])
      plotScalarCombined(avgTrainAcc, avgTestAcc, num_epochs, 'Tasa de Acierto Promedio', 'Tasa de Acierto', fig2, 3, 2, 5, lastEpoch)
      plotScalarCombined(avgTrainLost, avgTestLost, num_epochs, 'Error Promedio', 'Error', fig2, 3, 2, 6, lastEpoch)
    else:
      acc = np.max(test_acc[0:lastEpoch+1])
      plotScalarCombined(train_acc, test_acc, num_epochs, 'Tasa de Acierto Promedio', 'Tasa de Acierto', fig2, 3, 2, 5, lastEpoch)
      plotScalarCombined(train_lost, test_lost, num_epochs, 'Error Promedio', 'Error',fig2,3,2,6, lastEpoch)

    # plt.axvline(x=lastEpoch, color='g', linestyle='--')
    
    plt.text(0.0, 0.0, 'Accuracy: '+str(acc), horizontalalignment='center', verticalalignment='center',
          bbox=dict(boxstyle="square",
              ec=(1., 0.5, 0.5),
              fc=(1., 0.8, 0.8),))

    plt.show()
    print('max test accuracy until ',lastEpoch,' epoch: ', acc)
    # print('max test accuracy until ',lastEpoch,' epoch: ', np.amax(np.array(avgTestAcc[0:lastEpoch])))

def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learningCurvesFolder', type=str, help='Directory containing results')
    parser.add_argument('--modelName', type=str, help='model name')
    parser.add_argument('--lastEpoch', type=int, default=5, help='last epoch before overfiting')
    parser.add_argument('--numFolds', type=int, default=5)
    args = parser.parse_args()
    model_name = args.modelName
    path = os.path.join(args.learningCurvesFolder,model_name)
    lastEpoch = args.lastEpoch
    nfolds = args.numFolds
    plot_results(path, lastEpoch, nfolds)

__main__()