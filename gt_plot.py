import cv2
import sys
import argparse
import constants
import os
import numpy as np

def plotVideo(video_path, bdx_file_path, delay):
        data = []
        if bdx_file_path is not None:
            with open(bdx_file_path, 'r') as file:
                for row in file:
                    data.append(row.split())
            data = np.array(data)
        segment_info = []
        num_frame = 0
        magenta = (255, 50, 255)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
     
        while(cap.isOpened()):
            ret, frame = cap.read()
            
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if num_frame != int(data[num_frame, 5]):
                sys.exit('Houston we have a problem: index frame does not equal to the bbox file!!!')
            
            flac = int(data[num_frame,6]) # 1 if is occluded: no plot the bbox
            xmin = int(data[num_frame, 1])
            ymin= int(data[num_frame, 2])
            xmax = int(data[num_frame, 3])
            ymax = int(data[num_frame, 4])
            print(num_frame, flac,'(', xmin, ymin,xmax, ymax,')')
            if flac != 1:
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), magenta, 4)
            num_frame += 1
            cv2.imshow('frame',frame)
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

def plotVideoTest(video_path, delay):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    # print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps), video_path)
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame',frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

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
    cap.release()
    cv2.destroyAllWindows()

def waqasVideos2Frames(videos_folder, path_txt_videos, output_path):
    rows = []
    with open(path_txt_videos, 'r') as file:
          for row in file:
              rows.append(row)
    
    for row in rows:
        splits = row.split('/')
        video_name = splits[1]
        video_name = video_name[:-10]
        folderName = splits[0]
        # print(video_name)
        video_path = os.path.join(videos_folder,folderName,video_name+'_x264.mp4')
        video_out_path = os.path.join(output_path,video_name)
        # print(video_out_path)
        if not os.path.exists(video_out_path):
            os.makedirs(video_out_path)
        # plotVideoTest(video_path, 1)
        video2Images2(video_path,video_out_path)
        

def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str)
    parser.add_argument("--delay", type=int)
    args = parser.parse_args()
    video_name = args.video
    delay = args.delay
    # plotVideo(os.path.join(constants.PATH_UCFCRIME2LOCAL_VIDEOS, video_name + '_x264.mp4'), os.path.join(constants.PATH_UCFCRIME2LOCAL_BBOX_ANNOTATIONS, video_name + '.txt'),delay)
    # video_name = '/media/david/datos/Violence DATA/AnomalyCRIMEDATASET/Anomaly-Videos-All/Vandalism/' + video_name
    # plotVideoTest(video_name, delay)
    waqas_path = '/Volumes/TOSHIBA EXT/AnomalyCRIME'
    waqasVideos2Frames(waqas_path+'/Anomaly-Videos-All', 'Anomaly_Test.txt',
                        os.path.join('AnomalyCRIMEDATASET', 'waqas/test'))
    # video_name = os.path.join(waqas_path,'Anomaly-Videos-All',video_name)
    # for i in range(5):
    # plotVideoTest(video_name, delay)

__main__()