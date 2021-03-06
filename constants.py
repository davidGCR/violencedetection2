import os
from include import dirname, root
import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

PATH_HOCKEY_FRAMES_VIOLENCE = root+'/DATASETS/HockeyFightsDATASET/frames/violence'
PATH_HOCKEY_FRAMES_NON_VIOLENCE = root + '/DATASETS/HockeyFightsDATASET/frames/nonviolence'
PATH_HOCKEY_README = root + '/DATASETS/HockeyFightsDATASET/readme'
PATH_HOCKEY_VIDEOS= root+'/DATASETS/HockeyFightsDATASET/videos'
# PATH_HOCKEY_CHECKPOINTS = root + '/RESULTS/HOCKEY_RESULTS/checkpoints'
# PATH_HOCKEY_GIFTS = root + '/RESULTS/HOCKEY_RESULTS/gifts'

PATH_VIF_VIDEOS = root + '/DATASETS/violentflows/movies'
PATH_VIF_FRAMES = root + '/DATASETS/violentflows/frames'
PATH_VIF_README = root + '/DATASETS/violentflows/readme'

PATH_RWF_2000_VIDEOS = root + '/DATASETS/RWF-2000/videos'
PATH_RWF_2000_FRAMES = root + '/DATASETS/RWF-2000/frames'
PATH_RWF_2000_README = root + '/DATASETS/RWF-2000/readme'
PATH_RWF_2000_ROIS = root + '/DATASETS/RWF-2000/rois'

# import sys
# IN_COLAB = 'google.colab' in sys.modules
PATH_RESULTS = ''
if dirname == '/content':
    PATH_RESULTS = "/content/drive/My Drive/VIOLENCE DATA/RESULTS"
else:
    PATH_RESULTS = root + '/RESULTS'

# PATH_UCFCRIME2LOCAL_VIDEOS = root+'Crime2LocalDATASET/UCFCrime2Local/videos'
PATH_UCFCRIME2LOCAL_FRAMES_VIOLENCE = root + '/DATASETS/CrimeViolence2LocalDATASET/frames/violence'
PATH_UCFCRIME2LOCAL_FRAMES_NONVIOLENCE = root + '/DATASETS/CrimeViolence2LocalDATASET/frames/nonviolence'
PATH_UCFCRIME2LOCAL_README = root+'/DATASETS/CrimeViolence2LocalDATASET/readme'
PATH_UCFCRIME2LOCAL_Txt_ANNOTATIONS = root + '/DATASETS/CrimeViolence2LocalDATASET/Txt annotations-longVideos'
PATH_UCFCRIME2LOCAL_BBOX_ANNOTATIONS = root+'/DATASETS/CrimeViolence2LocalDATASET/bboxFiles'



PATH_VIOLENCECRIME2LOCAL_VIOLENCE_TRAIN_SPLIT = root + '/CrimeViolence2LocalDATASET/readme/Train_violence_split.txt'
PATH_VIOLENCECRIME2LOCAL_VIOLENCE_TEST_SPLIT = root + '/CrimeViolence2LocalDATASET/readme/Test_violence_split.txt'

PATH_VIOLENCECRIME2LOCAL_RAW_NONVIOLENCE_TRAIN_SPLIT = root + '/CrimeViolence2LocalDATASET/readme/Train_raw_nonviolence_split.txt'
PATH_VIOLENCECRIME2LOCAL_NEW_NONVIOLENCE_TRAIN_SPLIT = root + '/CrimeViolence2LocalDATASET/readme/Train_new_nonviolence_split.txt'
PATH_VIOLENCECRIME2LOCAL_RAW_NONVIOLENCE_TEST_SPLIT = root + '/CrimeViolence2LocalDATASET/readme/Test_raw_nonviolence_split.txt'
PATH_VIOLENCECRIME2LOCAL_NEW_NONVIOLENCE_TEST_SPLIT = root + '/CrimeViolence2LocalDATASET/readme/Test_new_nonviolence_split.txt'

PATH_FINAL_RANDOM_NONVIOLENCE_TRAIN_SPLIT = root + '/CrimeViolence2LocalDATASET/readme/Train_random_nonviolence_split.txt'
PATH_FINAL_RANDOM_NONVIOLENCE_TEST_SPLIT = root + '/CrimeViolence2LocalDATASET/readme/Test_random_nonviolence_split.txt'

PATHS_SPLITS_DICT = {'train_violence': PATH_VIOLENCECRIME2LOCAL_VIOLENCE_TRAIN_SPLIT,
                        'test_violence': PATH_VIOLENCECRIME2LOCAL_VIOLENCE_TEST_SPLIT,
                        'train_raw_nonviolence': PATH_VIOLENCECRIME2LOCAL_RAW_NONVIOLENCE_TRAIN_SPLIT,
                        'train_new_nonviolence': PATH_VIOLENCECRIME2LOCAL_NEW_NONVIOLENCE_TRAIN_SPLIT,
                        'test_raw_nonviolence': PATH_VIOLENCECRIME2LOCAL_RAW_NONVIOLENCE_TEST_SPLIT,
                        'test_new_nonviolence': PATH_VIOLENCECRIME2LOCAL_NEW_NONVIOLENCE_TEST_SPLIT}


# PATH_UCFCRIME2LOCAL_FRAMES_REDUCED = root+'Crime2LocalDATASET/frames_reduced'

# ANOMALY_PATH_CHECKPOINTS = root + '/ANOMALY_RESULTS/checkpoints'
# ANOMALY_PATH_GIFTS = root + '/ANOMALY_RESULTS/gifts'
# ANOMALY_PATH_CHECKPOINTS = root + '/drive/My Drive/VIOLENCE DATA/ANOMALY_RESULTS/checkpoints'

TEMP_MAX_POOL = 'maxTempPool'
TEMP_AVG_POOL = 'avgTempPool'
TEMP_STD_POOL = 'stdTempPool'
JOIN_CONCATENATE = 'cat'
MULT_TEMP_POOL = 'multipleTempPool'


# frame, flac, xmin, ymin, xmax, ymax
IDX_FRAME_NAME = 0
IDX_FLAC = 1
IDX_XMIN = 2
IDX_YMIN = 3
IDX_XMAX = 4
IDX_YMAX = 5
IDX_NUMFRAME = 6

RED = 'r'
GREEN = 'g'
CYAN = 'c'
PIL_WHITE = (255, 255, 255)
PIL_RED = (255, 0, 0)
PIL_YELLOW = (255, 255, 0)
PIL_GREEN = (0, 255, 0)
PIL_MAGENTA = (255, 0, 255)
PIL_BLUE = (0,0,255)

red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 0, 0)
yellow = (0, 255, 255)
magenta = (255,50,255)


YOLO = 'yolov3'
MASKRCNN = 'maskrcnn'