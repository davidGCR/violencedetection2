import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pickle
import glob
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import csv
from constants import DEVICE
import torch

def load_torch_checkpoint(path, model=None):
  checkpoint = torch.load(path, map_location=DEVICE)
  # model.load_state_dict(checkpoint['model_state_dict'])
  # if DEVICE == 'cuda:0':
  #     model.load_state_dict(checkpoint['model_state_dict'], strict=False)
  # else:
  #     model.load_state_dict(checkpoint['model_state_dict'], map_location=DEVICE))
  # epoch = checkpoint['epoch']
  # fold = checkpoint['fold']
  # val_acc = checkpoint['val_acc']
  # val_loss = checkpoint['val_loss']
  # model_config = checkpoint['model_config']

  # return model_config, epoch, val_acc, val_loss, fold
  return checkpoint

def expConfig(**kwargs):
  dict_ = {
    'dataset':kwargs['dataset'],
    'modelType': kwargs['modelType'],
    'numDynamicImages': kwargs['numDynamicImages'],
    'segmentLength': kwargs['segmentLength'],
    'joinType': kwargs['joinType'],
    'frameSkip': kwargs['frameSkip'],
    'featureExtract': kwargs['featureExtract'],
    'overlap': kwargs['overlap'],
    'skipInitialFrames': kwargs['skipInitialFrames'],
  }
  return dict_

def load_model_inference(file, device):
    if str(device) == 'cpu':
        model = torch.load(file, map_location=torch.device('cpu'))
    else:
        model = torch.load(file)
        model = model.cuda()
    model = model.eval()
    return model

def min_max_normalize_tensor(img):
    # print("normalize:", img.size())
    _min = torch.min(img)
    _max = torch.max(img)
    # print("min:", _min.item(), ", max:", _max.item())
    return (img - _min) / (_max - _min)

def min_max_normalize_np(img):
    # print("normalize:", img.size())
    _min = np.amin(img)
    _max = np.amax(img)
    # print("min:", _min.item(), ", max:", _max.item())
    return (img - _min) / (_max - _min)

def createGifFromFrames(path):
  frames = os.listdir(path)
  frames.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
  images = []
  for frame in frames:
    img = Image.open(os.path.join(path,frame))
    images.append(img)
  head, tail = os.path.split(path)
  images[0].save(os.path.join(path,tail+'.gif'),
               save_all=True, append_images=images[1:], optimize=False, duration=60, loop=0)

def print_balance(train_y,name):
    tx = np.array(train_y)
    unique, counts = np.unique(tx, return_counts=True)
    print(name +'-balance: ', dict(zip(unique, counts)))



def read_file(file):
  names = []
  with open(file, 'r') as file:
      for row in file:
          names.append(row[:-1])
  return names

def save_file(data, out_file):
  print('saving ... ', out_file)
  with open(out_file, 'w') as output:
      for row in data:
          output.write(str(row) + '\n')

def save_csvfile_multicolumn(data, out_file):
  print('saving ... ', out_file)
  with open(out_file, 'w') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(data)

def read_csvfile_threecolumns(file):
  x = []
  y = []
  numFrames = []
  with open(file) as csvfile:
    readCSV = csv.reader(csvfile, delimiter='\t')
    for row in readCSV:
        x.append(row[0])
        y.append(int(row[1]))
        numFrames.append(int(row[2]))
  return x, y, numFrames

def read_csvfile_twoColumns(file):
  x = []
  y = []
  with open(file) as csvfile:
    readCSV = csv.reader(csvfile, delimiter='\t')
    for row in readCSV:
        x.append(row[0])
        y.append(int(row[1]))
  return x, y
    

#######################################################################################
################################# Videos to Frames ####################################
#######################################################################################

def video2Images2(video_path, path_out):
  cap = cv2.VideoCapture(video_path)
  if (cap.isOpened()== False):
    print("Error opening video stream or file: ", video_path)
  #   return 0
  index_frame = 1
  # print(video_path)
  while(cap.isOpened()):
      ret, frame = cap.read()
      if not ret:
      #   print('video can not read ...')
        break
      name = path_out+'/'+'frame' + str("{0:03}".format(index_frame)) + '.jpg'
      print ('Creating...' + name)
      cv2.imwrite(name, frame)
      index_frame += 1

def sortListByStrNumbers(lt):
  """
    Sort a list of numeric strings as '0', '1', '2'
  """
  lt.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
  return lt    

      
def hockeyVideos2Frames(path_videos, path_frames):
  listViolence = os.listdir(os.path.join(path_videos, 'violence'))
  listViolence.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
  listnonViolence = os.listdir(os.path.join(path_videos, 'nonviolence'))
  listnonViolence.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
  
  for idx,video in enumerate(listViolence):
    path_video = os.path.join(path_videos, 'violence', video)
    path_frames_out = os.path.join(path_frames, 'violence', str(idx+1))
    if not os.path.exists(path_frames_out):
        os.makedirs(path_frames_out)
    video2Images2(path_video, path_frames_out)

  for idx,video in enumerate(listnonViolence):
    path_video = os.path.join(path_videos, 'nonviolence', video)
    path_frames_out = os.path.join(path_frames, 'nonviolence', str(idx+1))
    if not os.path.exists(path_frames_out):
        os.makedirs(path_frames_out)
    video2Images2(path_video,path_frames_out)

def videos2ImagesFromKfols(path_videos, path_frames):
  list_folds = os.listdir(path_videos) ## [1 2 3 4 5]
  for fold in list_folds:
    violence_videos_path = path_videos+'/'+fold+'/Violence'
    nonviolence_videos_path = path_videos+'/'+fold+'/NonViolence'
    
    violence_videos_path_out = path_frames+'/'+fold+'/Violence'
    nonviolence_videos_path_out = path_frames+'/'+fold+'/NonViolence'
    
    violent_videos_paths = os.listdir(violence_videos_path)
    nonviolent_videos_paths = os.listdir(nonviolence_videos_path)
    
    for video in violent_videos_paths:
      frames_path_out = os.path.join(path_frames,fold,'Violence',os.path.splitext(video)[0])
      print(frames_path_out)
      if not os.path.exists(frames_path_out):
        os.makedirs(frames_path_out)
      video2Images2(os.path.join(violence_videos_path,video), frames_path_out)
      
    for video in nonviolent_videos_paths:
      frames_path_out = os.path.join(path_frames,fold,'NonViolence',os.path.splitext(video)[0])
      print(frames_path_out)
      if not os.path.exists(frames_path_out):
        os.makedirs(frames_path_out)
        
      video2Images2(os.path.join(nonviolence_videos_path,video), frames_path_out)


def dataset_statistisc(path):
  l = os.listdir(path)
  n_frames = []
  for video_folder in l:
    frames_list = os.listdir(os.path.join(path,video_folder))
    num_frames = len(frames_list)
    n_frames.append(num_frames)

  avg = np.average(np.array(n_frames))
  return avg

