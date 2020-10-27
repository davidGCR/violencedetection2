# import os
# import shutil
# import re
# import cv2
# import numpy as np
# import time

# class VideoDataPreparation():
#     def __init__(self, dataset_folder, all_videos_file, train_split_file, test_split_file, dataset_frames_folder, temporal_annotations_folder, categories):
#         self.dataset_folder = dataset_folder
#         self.all_videos_file = all_videos_file
#         self.test_split_file = test_split_file
#         self.train_split_file = train_split_file
#         self.dataset_frames_folder = dataset_frames_folder
#         self.temporal_annotations_folder = temporal_annotations_folder
#         self.categories = categories

#     def get_videos_from_txt(self):
#         """Read video names from txt file and return list of str"""
#         f = open(self.all_videos_file, "r")
#         lines = f.readlines()
#         f.close()
#         lines = [line.rstrip('\n') for line in lines]
#         return lines
    
#     def get_video_name(self, raw_name):
#         """Extract video name and number from strings like Stealing102_x264"""
#         # name = re.findall(r'\w+', raw_name)
#         name = re.findall(r'[\w\.-]+_', raw_name)
#         number = re.sub("\D", "", name[0])
#         name = name[0][:-4]
#         return name, number
#         # video[0:6]

    
#     def choose_videos_frames(self, other_dataset_folder, categories):
#         """ Copy selected videos from another dataset/folder, it could be by categories or not"""
#         # ds_videos = self.get_videos_from_txt()
#         ds_videos = os.listdir(other_dataset_folder)
#         dst_folder = self.dataset_frames_folder
#         for video_name in ds_videos:
#             print(video_name)
#             vid_path_src = os.path.join(other_dataset_folder, video_name)
#             vid_path_dst = os.path.join(dst_folder, video_name)
#             label = video_name[:-3]
#             if not os.path.exists(vid_path_src):
#                 print('File {} does not exist...'.format(video_name))
#             elif label in categories:
#                 shutil.copytree(vid_path_src, vid_path_dst)
          
#     def sortAlphaNumericArray(self, lista):

#         lista.sort(key=lambda f: int(re.sub('\D', '', f)))
#         return lista

#     def get_video_intervals(self, num_frames, data, margen):
#         normal_intervals = []
#         start = -1
#         end = -1
#         itvl = False

#         for i, row in enumerate(data):
#             num_frame = data[i, 5]
#             flac = int(data[i, 6])  # 1 if is lost: no plot the bbox
#             # print(flac, num_frame)
#             if flac == 1:
#                 if not itvl:
#                     start = int(num_frame)
#                     itvl = True
#                 else:
#                     end = int(num_frame)
#             elif start>-1 and end >-1:
#                 interval = [start, end]
#                 normal_intervals.append(interval)
#                 itvl = False
#                 start = -1
#                 end = -1
#             if i == int(len(data) - 1) and itvl:
#                 end = num_frames
#                 interval = [start, end]
#                 normal_intervals.append(interval)
#                 itvl = False
#                 start = -1
#                 end = -1
        
#         violence_intervals = []
#         for idx, interval in enumerate(normal_intervals):
#             if idx + 1 < len(normal_intervals):
#                 normal_intervals[idx][1] = normal_intervals[idx][1] - margen
#                 if normal_intervals[idx + 1][0] + margen < num_frames: #verifica si el aumento no sobrepasa el numero de frames
#                     normal_intervals[idx + 1][0] = normal_intervals[idx + 1][0] + margen
#                 else:
#                     normal_intervals[idx + 1][0] = num_frames
#                 s = normal_intervals[idx][1]+1
#                 # print(s)
#                 e = normal_intervals[idx + 1][0]-1
#                 inter = [s, e]
#                 violence_intervals.append(inter)
#         return violence_intervals, normal_intervals

#     def temporal_split_video(self, video_frames_folder_src, violence_folder_dst, nonviolence_folder_dst, tmp_annotation_file, margen):
#         """Split a large video using temporal anotation"""
#         data=[]
#         with open(tmp_annotation_file, 'r') as file:
#             for row in file:
#                 data.append(row.split())
#         data.pop(0)
#         data = np.array(data)
        
#         # print(data[2])
#         all_frames = os.listdir(video_frames_folder_src)
        
#         all_frames.sort(key=lambda f: int(re.sub('\D', '', f)))
#         # print(all_frames)
            
#         #     data = np.delete(data,0)
#         if len(all_frames) - len(data) == 1:
#             all_frames.pop(len(all_frames)-1)
#             # print('Alerta: anotaciones incompletas en: {}, all_frames: {}, tmp_anotacion: {}'.format(video_frames_folder_src,len(all_frames), len(data)))
#         if len(all_frames) != len(data):
#             print('Nooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo')
#         print('{}, all_frames: {}, tmp_anotacion: {}'.format(video_frames_folder_src,len(all_frames), len(data)))
#         violence_intervals, normal_intervals = self.get_video_intervals(len(all_frames), data, margen)
#         print('violence_intervals: ', video_frames_folder_src)
#         print(violence_intervals, normal_intervals)

#         _, video_name = os.path.split(video_frames_folder_src)
#         v_path = os.path.join(violence_folder_dst,video_name)
#         for idx,violence_interval in enumerate(violence_intervals):
#             path_violence_segment = v_path+'-VSplit'+str(idx+1)
#             if not os.path.exists(path_violence_segment):
#                 os.makedirs(path_violence_segment)
#                 i = violence_interval[0]
#                 while i <= violence_interval[1]:
#                     index = i-1
#                     frame_name = all_frames[index]
#                     frame_number = int(re.sub("\D", "", frame_name))
#                     if frame_number == i:
#                         shutil.copy(os.path.join(video_frames_folder_src,all_frames[index]), path_violence_segment)
#                         i += 1
#                     else:
#                         print('No coincidende: ', i, frame_number)
#                         break
#             # violence_paths.append(path_violence_segment)
#             # train_labels.append(1)
#         v_path = os.path.join(nonviolence_folder_dst,'Normal_Videos-'+video_name)
#         for idx,normal_interval in enumerate(normal_intervals):
#             path_normal_segment = v_path + '-NSplit-' + str(idx + 1)
#             # print(path_normal_segment)
#             if not os.path.exists(path_normal_segment):
#                 os.makedirs(path_normal_segment)
#                 i = normal_interval[0]
#                 while i <= normal_interval[1]:
#                     index = i - 1
#                     # print(index)
#                     frame_name = all_frames[index]
#                     frame_number = int(re.sub("\D", "", frame_name))
#                     if frame_number == i:
#                         shutil.copy(os.path.join(video_frames_folder_src,all_frames[index]), path_normal_segment)
#                         i += 1
#                     else:
#                         print('No coincidende: ', i, frame_number)
#                         break
#             # non_violence_paths.extend([path_normal_segment])

#     def split_all_videos(self,margen):
#         videos = os.listdir(os.path.join(self.dataset_frames_folder, 'violence'))
#         videos.sort()
#         for video in videos:
#             self.temporal_split_video(video_frames_folder_src=os.path.join(self.dataset_frames_folder, 'violence',video),
#                                         violence_folder_dst='CrimeViolence2LocalDATASET/frames/violence',
#                                         nonviolence_folder_dst='CrimeViolence2LocalDATASET/frames/nonviolence',
#                                         tmp_annotation_file=os.path.join(self.temporal_annotations_folder, video + '.txt'),
#                                         margen=margen)
#         # video = videos[5]
#         # self.temporal_split_video(video_frames_folder_src=os.path.join(self.dataset_frames_folder, 'violence',video),
#         #                                 violence_folder_dst=os.path.join(self.dataset_frames_folder, 'violence2'),
#         #                                 nonviolence_folder_dst=os.path.join(self.dataset_frames_folder, 'nonviolence2'),
#         #                                 tmp_annotation_file=os.path.join(self.temporal_annotations_folder, video + '.txt'),
#         #                                 margen=margen)
#     # def read_file(self, file):
#     #     names = []
#     #     with open(file, 'r') as file:
#     #         for row in file:
#     #             names.append(row[:-1])
#     #     return names

#     def save_file(self, data, out_file):
#         with open(out_file, 'w') as output:
#             for row in data:
#                 output.write(str(row) + '\n')

#     def generate_train_test_files(self, folders_path):
#         violence_folder=folders_path['violence_folder']
#         nonviolence_folder=folders_path['nonviolence_folder']
#         in_original_file=folders_path['in_original_file']
#         out_violence_file=folders_path['out_violence_file']
#         out_raw_nonviolence_file=folders_path['out_raw_nonviolence_file']
#         out_new_nonviolence_file=folders_path['out_new_nonviolence_file']
        
#         print(type(violence_folder))
#         all_frames_violence = os.listdir(violence_folder)
#         all_frames_violence.sort()
#         names = self.read_file(in_original_file)
#         new_violence_samples=[]
#         for folder in all_frames_violence:
#             video_r = re.findall(r'[\w\.-]+-', folder)
#             video_r = video_r[0][:-1]
#             if video_r in names:
#                 new_violence_samples.append(folder)
#         new_violence_samples.sort()
#         self.save_file(new_violence_samples,out_violence_file)

#         raw_nonviolence_samples = []
#         for name in names:
#             normal = name[:-3]
#             if normal == 'Normal_Videos':
#                 raw_nonviolence_samples.append(name)
#         raw_nonviolence_samples.sort()
#         self.save_file(raw_nonviolence_samples, out_raw_nonviolence_file)

#         all_frames_nonviolence = os.listdir(nonviolence_folder)
#         all_frames_nonviolence.sort()
#         new_nonviolence_samples=[]
#         for folder in all_frames_nonviolence:
#             video_r = re.findall(r'[\w\.-]+-', folder)
#             video_r = video_r[0][:-1]
#             new_nonviolence_samples.append(folder)
#         new_nonviolence_samples.sort()       
#         self.save_file(new_nonviolence_samples, out_new_nonviolence_file)

#     def frames2Video(self, folder_in, folder_out, fps):
#         frames = os.listdir(folder_in)
#         frames = self.sortAlphaNumericArray(frames)
#         frame_array = []
#         for i in range(len(frames)):
            
#             frame_path = os.path.join(folder_in, frames[i])
#             #reading each files
#             img = cv2.imread(frame_path)
#             height, width, layers = img.shape
#             size = (width,height)
#             print(frame_path)
#             #inserting the frames into an image array
#             frame_array.append(img)
#         h, videoname = os.path.split(folder_in)
#         videoname = videoname + '.avi'
#         videoPath = os.path.join(folder_out,videoname)
#         out = cv2.VideoWriter(videoPath,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

#         for i in range(len(frame_array)):
#             # writing to a image array
#             out.write(frame_array[i])
#         out.release()

# def getFrame(vidcap, sec, path_out):
#     vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
#     hasFrames,image = vidcap.read()
#     if hasFrames:
#         cv2.imwrite(path_out, image)     # save frame as JPG file
#     return hasFrames

# def video2Frames2(video_path, path_out, delayTime):
#     cap = cv2.VideoCapture(video_path)

#     _, videoName = os.path.split(video_path)
#     videoName = videoName[:-4]
#     framesPath = os.path.join(path_out, videoName)
#     if not os.path.exists(framesPath):
#         os.makedirs(framesPath)
    
#     sec = 0
#     frameRate = delayTime #it will capture image in each 0.5 second
#     index_frame = 1
#     frame_name = framesPath+'/'+'frame' + str("{0:04}".format(index_frame)) + '.jpg'
#     success = getFrame(cap, sec, frame_name)
#     while success:
#         index_frame += 1
#         sec = sec + frameRate
#         sec = round(sec, 2)
#         frame_name = framesPath+'/'+'frame' + str("{0:04}".format(index_frame)) + '.jpg'
#         success = getFrame(cap, sec, frame_name)
            
        
#     # def video2Frames(self,video_path, path_out, delayTime):
#     #     cap = cv2.VideoCapture(video_path)
#     #     cap.set(cv2.CAP_PROP_POS_MSEC, 20)
#     #     if (cap.isOpened()== False):
#     #         print("Error opening video stream or file: ", video_path)
#     #     #   return 0
#     #     index_frame = 1
#     #     # print(video_path)
#     #     _, videoName = os.path.split(video_path)
#     #     videoName = videoName[:-4]
#     #     framesPath = os.path.join(path_out, videoName)
#     #     if not os.path.exists(framesPath):
#     #         os.makedirs(framesPath)
#     #     while(cap.isOpened()):
#     #         ret, frame = cap.read()
#     #         if ret == False:
#     #             break
#     #         frame_name = framesPath+'/'+'frame' + str("{0:03}".format(index_frame)) + '.jpg'
#     #         print ('Creating...' + frame_name)
#     #         cv2.imwrite(frame_name, frame)
#     #         index_frame += 1
#     #         # cv2.waitKey(delayTime)
#     #         time.sleep(delayTime)


# def __main__():
#     # local_dataset = '/media/david/datos/Violence DATA/AnomalyCRIME/UCFCrime2Local/videos'
#     categories = {'Normal_Videos':0, 'Arrest': 1, 'Assault': 2, 'Robbery': 4, 'Stealing': 5}
#     dataPreparator = VideoDataPreparation(dataset_folder='Crime2LocalDATASET/videos',
#                                             all_videos_file='Crime2LocalDATASET/readme/Videos_from_UCFCrime.txt',
#                                             train_split_file='Crime2LocalDATASET/readme/Train_split_AD.txt',
#                                             test_split_file='Crime2LocalDATASET/readme/Test_split_AD.txt',
#                                             dataset_frames_folder='Crime2LocalDATASET/frames',
#                                             temporal_annotations_folder='Crime2LocalDATASET/Txt annotations', 
#                                             categories=categories)
#     # lines = dataPreparator.get_videos_from_txt()
#     # print(len(lines))
#     # print(lines)
#     # dataPreparator.choose_videos_frames(other_dataset_folder='Crime2LocalDATASET/frames', categories=categories)
#     # nn = re.sub("\D", "", 'Vandalism019')
#     # print(int(nn))
#     # frame = 'frame005.jpg'
#     # num_frame = int(frame[len(frame) - 7:-4])
#     # print(num_frame)
#     # dataPreparator.split_all_videos(margen=10)

#     folders_path = {'violence_folder': "CrimeViolence2LocalDATASET/frames/violence",
#                     'nonviolence_folder': "CrimeViolence2LocalDATASET/frames/nonviolence",
#                     'in_original_file': "Crime2LocalDATASET/readme/Train_split_AD.txt",
#                     'out_violence_file': "CrimeViolence2LocalDATASET/readme/Train_violence_split.txt",
#                     'out_raw_nonviolence_file': "CrimeViolence2LocalDATASET/readme/Train_raw_nonviolence_split.txt",
#                     'out_new_nonviolence_file': "CrimeViolence2LocalDATASET/readme/Train_new_nonviolence_split.txt"
#                     }
#     # dataPreparator.generate_train_test_files(folders_path)
#     folder_in = "/Users/davidchoqueluqueroman/Desktop/PAPERS-CODIGOS/violencedetection2/CrimeViolence2LocalDATASET/frames/violence/Stealing055-VSplit2"
#     folder_out = '/Users/davidchoqueluqueroman/Desktop/PAPERS-CODIGOS/violencedetection2/CrimeViolence2LocalDATASET/videos'
#     fps = 30
#     dataPreparator.frames2Video(folder_in, folder_out, fps)
#     # videoPath = '/Users/davidchoqueluqueroman/Desktop/PAPERS-CODIGOS/violencedetection2/CrimeViolence2LocalDATASET/videos/Stealing055-VSplit2.avi'
#     # framesPath = "/Users/davidchoqueluqueroman/Desktop/PAPERS-CODIGOS/violencedetection2/CrimeViolence2LocalDATASET/frames20fps"
#     # delayTime = 0.5
#     # # video2Frames2(videoPath, framesPath, delayTime)
#     # dataPreparator.frames2Video(framesPath+'/'+'Stealing055-VSplit2', framesPath, fps)
#     # vv = os.listdir('CrimeViolence2LocalDATASET/frames/nonviolence')



# __main__()