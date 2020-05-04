import os
from include import *
# root = '/Users/davidchoqueluqueroman/Desktop/PAPERS-CODIGOS/violencedetection2/'
PATH_DATA_AUMENTATION_OUTPUT = root+'/AnomalyCRIMEDATASET/UCFCrime2Local/dataAumentation'
PATH_HOCKEY_FRAMES_VIOLENCE = root+'/HockeyFightsDATASET/frames/violence'
PATH_HOCKEY_FRAMES_NON_VIOLENCE = root + '/HockeyFightsDATASET/frames/nonviolence'
PATH_HOCKEY_README = root + '/HockeyFightsDATASET/readme'
HOCKEY_PATH_CHECKPOINTS = root + '/HOCKEY_RESULTS/checkpoints'
# HOCKEY_PATH_CHECKPOINTS = root + '/drive/My Drive/VIOLENCE DATA/HOCKEY_RESULTS/checkpoints'

PATH_HOCKEY_AUMENTED_VIOLENCE = root + '/HockeyFightsDATASET/dataAumentation/violence'
PATH_HOCKEY_AUMENTED_NON_VIOLENCE = root + '/HockeyFightsDATASET/dataAumentation/nonviolence'

PATH_VIOLENTFLOWS_FRAMES = root+'/violentflows/movies Frames'
PATH_VIOLENCE_CHECKPOINTS = 'VIOLENCE_RESULTS/checkpoints'
PATH_TIME_RESULTS = 'VIOLENCE_RESULTS/timeResults'
PATH_VIOLENCE_ROC_CURVES = 'VIOLENCE_RESULTS/roc_curves'
PATH_VIOLENCE_TMP_RESULTS = 'VIOLENCE_RESULTS/temporalResults'
PATH_VIOLENCE_TMP_PLOTS = 'VIOLENCE_RESULTS/temporalPlots'
# PATH_CHECKPOINTS_MASK = 'checkpoints/masked'
PATH_VIOLENCE_LEARNING_CURVES = 'VIOLENCE_RESULTS/learningCurves'
# PATH_LEARNING_CURVES_MASK = 'learningCurves/masked'
PATH_SALIENCY_MODELS = 'SALIENCY/Models'
PATH_BLACK_BOX_MODELS = 'BlackBoxModels'
PATH_SALIENCY_DATASET = 'DATA/saliency'
# PATH_UCFCRIME2LOCAL_VIDEOS = root+'Crime2LocalDATASET/UCFCrime2Local/videos'
PATH_UCFCRIME2LOCAL_FRAMES_VIOLENCE = root + '/CrimeViolence2LocalDATASET/frames/violence'
PATH_UCFCRIME2LOCAL_FRAMES_NEW_NONVIOLENCE = root + '/CrimeViolence2LocalDATASET/frames/nonviolence'
PATH_UCFCRIME2LOCAL_FRAMES_RAW_NONVIOLENCE = root + '/Crime2LocalDATASET/frames/nonviolence'


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

PATH_VIOLENCECRIME2LOCAL_README = root+'/CrimeViolence2LocalDATASET/readme'
PATH_VIOLENCECRIME2LOCAL_BBOX_ANNOTATIONS = root+'/CrimeViolence2LocalDATASET/Txt annotations'
# PATH_UCFCRIME2LOCAL_FRAMES_REDUCED = root+'Crime2LocalDATASET/frames_reduced'

ANOMALY_PATH_CHECKPOINTS = root + '/ANOMALY_RESULTS/checkpoints'
# ANOMALY_PATH_CHECKPOINTS = root + '/drive/My Drive/VIOLENCE DATA/ANOMALY_RESULTS/checkpoints'

ANOMALY_PATH_LEARNING_CURVES = root + '/ANOMALY_RESULTS/learning_curves'
ANOMALY_PATH_TRAIN_SPLIT = os.path.join(PATH_VIOLENCECRIME2LOCAL_README, 'Train_split_AD.txt')
ANOMALY_PATH_TEST_SPLIT = os.path.join(PATH_VIOLENCECRIME2LOCAL_README, 'Test_split_AD.txt')
ANOMALY_PATH_SALIENCY_MODELS = 'SALIENCY/Models/anomaly'
ANOMALY_PATH_BLACK_BOX_MODELS = 'SALIENCY/BlackBoxModels/anomaly'

OPERATION_TRAINING = 'train'
OPERATION_TRAINING_FINAL = 'trainfinal'
OPERATION_TESTING = 'testing'
OPERATION_TRAINING_AUMENTED = 'aumented'
OPERATION_TRAINING_TRANSFER = 'transfer'

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

FRAME_POS_FIRST = 'first'
FRAME_POS_END = 'end'
FRAME_POS_ALL = 'all'
FRAME_POS_EXTREMES = 'extremes'

YOLO = 'yolov3'
MASKRCNN = 'maskrcnn'