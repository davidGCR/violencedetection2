import pickle
import os
import glob
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

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

def get_model_name(modelType, scheduler_type, numDiPerVideos, feature_extract, joinType,num_epochs):
    model_name = str(modelType) + '_Finetuned-' + str(not feature_extract) + '-' +'_di-'+str(numDiPerVideos) + '_fusionType-'+str(joinType) +'_num_epochs-' +str(num_epochs)
    return model_name
    
def save_checkpoint(state, path):
  print('saving checkpoint ...')
  torch.save(state, path+'.tar')



def imshow(img, title):
    img = img / 2 + 0.5    
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = True



def saveList(path_out,model, scheduler_type,curve, numDI, feature_extract, joinType,lista):
  data_file = path_out+'/'+str(model)+'-Finetuned:'+str(not feature_extract)+'-'+str(numDI)+'di-'+joinType+'-'+scheduler_type+'-'+str(curve)+'.txt'
  with open(data_file, 'wb') as filehandle:
      # store the data as binary data stream
    pickle.dump(lista, filehandle)
    print('saved ... ', data_file)

def saveLearningCurve(path_out, lista):
  data_file = path_out
  with open(data_file, 'wb') as filehandle:
      # store the data as binary data stream
    pickle.dump(lista, filehandle)
    print('saved ... ',data_file)

def saveData(path_out,model,curve, numDI, source_type, feature_extract, joinType,lista):
  data_file = path_out+'/'+str(model)+'-'+source_type+'-Finetuned:'+str(not feature_extract)+'-'+str(numDI)+'di-'+joinType+'-'+str(curve)+'.txt'
  with open(data_file, 'wb') as filehandle:
      # store the data as binary data stream
    pickle.dump(lista, filehandle)
    print('saved ... ', data_file)
    
def saveList2(path_out,lista):
  data_file = path_out
  with open(data_file, 'wb') as filehandle:
      # store the data as binary data stream
    pickle.dump(lista, filehandle)
    print('saved ... ',data_file)

def loadList(name):
  with open(name, 'rb') as filehandle:
    # read the data as binary data stream
    hist2 = pickle.load(filehandle)
    return hist2

def loadArray(path_out, model,curve, numDI, source_type, feature_extract, joinType):
  data_file = path_out+'/'+str(model)+'-'+source_type+'-Finetuned:'+str(not feature_extract)+'-'+str(numDI)+'di-'+joinType+'-'+str(curve)+'.txt'
  with open(data_file, 'rb') as filehandle:
    # read the data as binary data stream
    print('loading... ',data_file)
    hist2 = pickle.load(filehandle)
    return hist2

def saveArray(path, array):
  with open(path,'wb') as f:
    print('saved ... ',path)
    pickle.dump(array, f)



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
      # if not ret:
      #   print('video can not read ...')
      #   break
      name = path_out+'/'+'frame' + str("{0:03}".format(index_frame)) + '.jpg'
      print ('Creating...' + name)
      cv2.imwrite(name, frame)
      index_frame += 1
          

      
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