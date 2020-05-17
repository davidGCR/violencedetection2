import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch 
import pandas as pd
import os
import argparse
from util import *
import constants
from sklearn.metrics import roc_curve, roc_auc_score, auc

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

def plot_results(path, lastEpoch, nfolds,title, mode):
    train_lost = loadList(str(path)+'-train_lost.txt')
    train_acc = loadList(str(path) + '-train_acc.txt')
    # if mode=='train':
    if mode=='val':
      test_lost = loadList(str(path)+'-val_lost.txt')
      test_acc = loadList(str(path)+'-val_acc.txt')

    num_epochs = int(len(train_lost)/nfolds)    
    acc = 0
    if nfolds == 5:
      fig2 = plt.figure(figsize=(12,12))
      rows = 3
      cols = 2
      plotScalarFolds(train_acc,train_lost,num_epochs,'Train',fig2,rows,cols,1)
      plotScalarFolds(test_acc, test_lost, num_epochs, 'Val',fig2,rows,cols,3)
      avgTrainAcc = getAverageFromFolds(train_acc,num_epochs)
      avgTrainLost = getAverageFromFolds(train_lost,num_epochs)
      avgTestAcc = getAverageFromFolds(test_acc,num_epochs)
      avgTestLost = getAverageFromFolds(test_lost, num_epochs)
      acc = np.max(avgTestAcc[0:lastEpoch+1])
      plotScalarCombined(avgTrainAcc, avgTestAcc, num_epochs, 'Tasa de Acierto Promedio', 'Tasa de Acierto', fig2, rows, cols, 5, lastEpoch)
      plotScalarCombined(avgTrainLost, avgTestLost, num_epochs, 'Error Promedio', 'Error', fig2, rows, cols, 6, lastEpoch)
      plt.text(0.0, 0.0, 'Accuracy: '+str(acc), horizontalalignment='center', verticalalignment='center',
            bbox=dict(boxstyle="square",
            ec=(1., 0.5, 0.5),
            fc=(1., 0.8, 0.8),))
    else:
      fig2 = plt.figure(figsize=(15, 4))
      fig2.suptitle(title, fontsize=14)
      rows = 1
      cols = 2
      
      if mode == 'train':
        acc = 0
        plotScalarCombined(train_acc, train_acc, num_epochs, 'Curva de Tasa de Acierto', 'Tasa de Acierto', fig2, rows, cols, 1, lastEpoch)
        plotScalarCombined(train_lost, train_lost, num_epochs, 'Curva de Error', 'Error',fig2,rows,cols,2, lastEpoch)
      else:
        acc = np.max(test_acc[0:lastEpoch + 1])
        plotScalarCombined(train_acc, test_acc, num_epochs, 'Curva de Tasa de Acierto', 'Tasa de Acierto', fig2, rows, cols, 1, lastEpoch)
        plotScalarCombined(train_lost, test_lost, num_epochs, 'Curva de Error', 'Error',fig2,rows,cols,2, lastEpoch)

    # plt.axvline(x=lastEpoch, color='g', linestyle='--')
      # y=test_acc[lastEpoch].cpu().numpy()
      # plt.scatter(lastEpoch, y, marker='x', color='red')
      # plt.text(lastEpoch+.03, y, str(acc), fontsize=9)
      # plt.text(0.5, 0.25, 'Accuracy: '+str(acc), horizontalalignment='center', verticalalignment='center',
      #       bbox=dict(boxstyle="square",
      #           ec=(1., 0.5, 0.5),
      #           fc=(1., 0.8, 0.8),))

    plt.show()
    print('max test accuracy until ',lastEpoch,' epoch: ', acc)

def plotROCCurve(tpr, fpr):
  fig = plt.figure()
  ax=fig.gca()
  vauc = auc(fpr, tpr)  #type 2  
  plt.plot(fpr, tpr, lw=1, color='red', marker='.', label='(AUC = %0.4f)' % (vauc))

  # plt.plot(fpr, tpr, lw=1, color='darkorange', marker='.', label='Curva ROC (area = %0.4f)' % vauc)
  # plt.plot(fpr2, tpr2, lw=1, color='red', marker='.', label='Curva ROC (area = %0.4f)' % vauc2)
  plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
  plt.xlim(xmin=0.0, xmax=1)
  plt.ylim(ymin=0.0, ymax=1)
  ax.set_xticks(np.arange(0,1,0.1))
  ax.set_yticks(np.arange(0,1,0.1))
  # plt.scatter()
  plt.grid()
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('ROC')
  plt.legend(loc="lower right")
  plt.show()

def plotROCCurvesFromFiles(path):
  # your_data = {'videoBlockLength': 70,
  #               'BlockOverlap':0,
  #               'videoSegmentLength': 20,
  #               'SegmentOverlap': 0.5,
  #               'model': 'testresnet50-6-Finetuned:True-maxTempPool-numEpochs:9-videoSegmentLength:20-overlaping:0.5-only_violence:True',
  #               'tpr': tpr,
  #               'fpr': fpr}
  
  # pickle.dump(your_data, open(os.path.join(constants.PATH_VIOLENCE_ROC_CURVES,"variation1.pkl"), "wb"))
  vars_experiments = os.listdir(path)
  vars_experiments.sort()
  # vars_experiments.sort(key=lambda x:os.path.getmtime(x))
  colors = ['blue', 'red', 'black', 'cyan', 'magenta', 'grey', 'purple', 'yellow', 'lime']
  fig = plt.figure()
  ax=fig.gca()
  for i, exp in enumerate(vars_experiments):

    var1 = pickle.load(open(os.path.join(path, exp), "rb"))
    # info = 'BLen:%d-BOvlp:%.2f-SLen:%d-SOvlp:%.2ff-BlNumDis:%d'%(var1['videoBlockLength'],var1['BlockOverlap'],var1['videoSegmentLength'],var1['SegmentOverlap'],var1['numDynamicImgsPerBlock'])
    info = 'bl=%d, bo=%.2f'%(var1['videoBlockLength'],var1['BlockOverlap'])
    # info = ''
    model = var1['model']
    print(colors[i], model)

    tpr = var1['tpr']
    fpr = var1['fpr']
    vauc = auc(fpr, tpr)  #type 2  
    plt.plot(fpr, tpr, lw=1, color=colors[i], marker='.', label='%s (AUC = %0.4f)' % (info,vauc))

  # plt.plot(fpr, tpr, lw=1, color='darkorange', marker='.', label='Curva ROC (area = %0.4f)' % vauc)
  # plt.plot(fpr2, tpr2, lw=1, color='red', marker='.', label='Curva ROC (area = %0.4f)' % vauc2)
  plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
  plt.xlim(xmin=0.0, xmax=1)
  plt.ylim(ymin=0.0, ymax=1)
  ax.set_xticks(np.arange(0,1,0.1))
  ax.set_yticks(np.arange(0,1,0.1))
  # plt.scatter()
  plt.grid()
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('ROC')
  plt.legend(loc="lower right")
  plt.show()

def plot_temporal_results(video_dict, threshold, save):
  v_name = video_dict['name']
  video_y_gt = video_dict['y_truth']
  video_y_pred = video_dict['y_pred_score']
  video_y_pred_class = video_dict['y_pred_class']
  x_axis = np.arange(0, len(video_y_gt), 1)
  thresh_line = threshold*np.ones(len(video_y_gt))

  # for i, label in enumerate(video_y_pred_class):
  #   if video_y_pred[i] >= 0.6:
  #     video_y_pred_class[i] = 1
  #   else:
  #     video_y_pred_class[i] = 0
  fig = plt.figure(figsize=(10,4))
  plt.plot(x_axis, video_y_pred,color='red', label='score',linewidth=2)
  # plt.plot(x_axis, video_y_pred_class, color='red', label='binary class', linewidth=1)
  plt.plot(x_axis, thresh_line, '--',color='navy', label='threshold line')
  # plt.fill_between(x_axis, video_y_pred_class, color='lightcoral')
  

  plt.plot(x_axis, video_y_gt, color='black', label='ground-truth', linewidth=2)
  plt.fill_between(x_axis, video_y_gt, color='dimgrey')
  # plt.bar(x_axis, video_y_pred_class, hatch='x', color='dimgrey')
  
  plt.xlim(xmin=0, xmax=len(video_y_gt))
  plt.ylim(ymin=0, ymax=1)
  plt.xlabel('Frame number')
  plt.ylabel('Score')
  plt.title(v_name)
  plt.legend(loc="lower right")
  if save:
    # h, t = os.path.split(path_video)
    plt.savefig(os.path.join(constants.PATH_VIOLENCE_TMP_PLOTS,v_name+'.png'))
  else:
    plt.show()
  plt.close(fig)

def plot_temporal_results_fromFile(path_video, path_out,threshold, save):
  video = pickle.load(open(path_video, "rb"))
  v_name = video['name']
  video_y_gt = video['y_truth']
  video_y_pred = video['y_pred_score']
  video_y_pred_class = video['y_pred_class']
  x_axis = np.arange(0, len(video_y_gt), 1)
  thresh_line = threshold*np.ones(len(video_y_gt))

  # for i, label in enumerate(video_y_pred_class):
  #   if video_y_pred[i] >= 0.6:
  #     video_y_pred_class[i] = 1
  #   else:
  #     video_y_pred_class[i] = 0
  fig = plt.figure()
  plt.plot(x_axis, video_y_pred, color='magenta', label='score',linewidth=2)
  plt.plot(x_axis, video_y_pred_class, color='red', label='binary class', linewidth=1)
  plt.plot(x_axis, thresh_line, '--',color='navy', label='threshold line')
  plt.fill_between(x_axis, video_y_pred_class, color='lightcoral')
  

  plt.plot(x_axis, video_y_gt, color='black', label='ground-truth', linewidth=2)
  plt.fill_between(x_axis, video_y_gt, color='dimgrey')
  # plt.bar(x_axis, video_y_pred_class, hatch='x', color='dimgrey')
  
  plt.xlim(xmin=0, xmax=len(video_y_gt))
  plt.ylim(ymin=0, ymax=1)
  plt.xlabel('Frame number')
  plt.ylabel('Score')
  plt.title(v_name)
  plt.legend(loc="lower right")
  if save:
    # h, t = os.path.split(path_video)
    plt.savefig(os.path.join(path_out,v_name+'.png'))
  else:
    plt.show()


def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plotMultipleRoc', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--tmpPlotVideo', type=str, default='')
    parser.add_argument('--plotRocCurve', type=str, default='')
    parser.add_argument('--learningCurve', type=str, default='')
    parser.add_argument('--lastEpoch', type=int, default=5)
    parser.add_argument('--nFolds',type=int, default=1)
    parser.add_argument('--mode', type=str, default='train')

    args = parser.parse_args()
    # tmpPlotsAllDataset = args.tmpPlotsAllDataset
    tmpPlotVideo = args.tmpPlotVideo
    plotRocCurve = args.plotRocCurve
    plotMultipleRoc = args.plotMultipleRoc
    learningCurve = args.learningCurve
    mode = args.mode
    nfolds = args.nFolds
    lastEpoch = args.lastEpoch

    # plotROCCurves(constants.PATH_VIOLENCE_ROC_CURVES+'/dicts')
    if plotMultipleRoc:
      plotROCCurvesFromFiles(constants.PATH_VIOLENCE_ROC_CURVES+'/dicts')
      # videos_test = os.listdir(constants.PATH_VIOLENCE_TMP_RESULTS)
      # videos_test.sort()
      # for video in videos_test:
      #   print(video)
      #   plot_temporal_results_fromFile(os.path.join(constants.PATH_VIOLENCE_TMP_RESULTS, video), path_out=constants.PATH_VIOLENCE_TMP_PLOTS, threshold=0.5, save=True)
    elif plotRocCurve != '':
      var1 = pickle.load(open(plotRocCurve, "rb"))
      # info = ''
      model = var1['model']
      tpr = var1['tpr']
      fpr = var1['fpr']
      plotROCCurve(tpr, fpr)
    elif learningCurve != '':
      plot_results(learningCurve, lastEpoch=lastEpoch, nfolds=nfolds,title=learningCurve, mode=mode)

    else:
      plot_temporal_results_fromFile(os.path.join(constants.PATH_VIOLENCE_TMP_RESULTS, tmpPlotVideo+'.pkl'), path_out='', threshold=0.5, save=False)


if __name__ == '__main__':
    __main__()