import os
# root = '/media/david/datos/Violence DATA/'
root = '/Users/davidchoqueluqueroman/Desktop/PAPERS-CODIGOS/violencedetection2/'
PATH_HOCKEY_FRAMES_VIOLENCE = root+'HockeyFights/frames/violence'
PATH_HOCKEY_FRAMES_NON_VIOLENCE = root+'HockeyFights/frames/nonviolence'
PATH_VIOLENTFLOWS_FRAMES = root+'violentflows/movies Frames'
PATH_CHECKPOINTS_DI = 'checkpoints/di'
PATH_CHECKPOINTS_MASK = 'checkpoints/masked'
PATH_LEARNING_CURVES_DI = 'learningCurves/di'
PATH_LEARNING_CURVES_MASK = 'learningCurves/masked'
PATH_SALIENCY_MODELS = 'SALIENCY/Models'
PATH_BLACK_BOX_MODELS = 'BlackBoxModels'
LABEL_PRODUCTION_MODEL = '-PRODUCT'
PATH_SALIENCY_DATASET = 'DATA/saliency'
PATH_UCFCRIME2LOCAL_VIDEOS = root+'AnomalyCRIMEDATASET/UCFCrime2Local/videos'
PATH_UCFCRIME2LOCAL_FRAMES = root+'AnomalyCRIMEDATASET/UCFCrime2Local/frames'
PATH_UCFCRIME2LOCAL_README = root+'AnomalyCRIMEDATASET/UCFCrime2Local/readme'
PATH_UCFCRIME2LOCAL_BBOX_ANNOTATIONS = root+'AnomalyCRIMEDATASET/UCFCrime2Local/readme/Txt annotations'
PATH_UCFCRIME2LOCAL_FRAMES_REDUCED = root+'AnomalyCRIMEDATASET/UCFCrime2Local/frames_reduced'

ANOMALY_PATH_CHECKPOINTS = 'AnomalyCRIMEDATASET/checkpoints'
ANOMALY_PATH_LEARNING_CURVES = 'AnomalyCRIMEDATASET/learning_curves'
ANOMALY_PATH_TRAIN_SPLIT = os.path.join(PATH_UCFCRIME2LOCAL_README, 'Train_split_AD.txt')
ANOMALY_PATH_TEST_SPLIT = os.path.join(PATH_UCFCRIME2LOCAL_README, 'Test_split_AD.txt')
ANOMALY_PATH_SALIENCY_MODELS = 'SALIENCY/Models/anomaly'
ANOMALY_PATH_BLACK_BOX_MODELS = 'SALIENCY/BlackBoxModels/anomaly'

OPERATION_TRAINING = 'train'
OPERATION_TRAINING_FINAL = 'trainfinal'
OPERATION_TESTING = 'testing'

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

RED = 'r'
GREEN = 'g'
CYAN = 'c'
PIL_WHITE = (255, 255, 255)
PIL_RED = (255, 0, 0)
PIL_YELLOW = (255, 255, 0)
PIL_GREEN = (0, 255, 0)
PIL_MAGENTA = (255, 0, 255)
PIL_BLUE = (0,0,255)

FRAME_POS_FIRST = 'first'
FRAME_POS_END = 'end'
FRAME_POS_ALL = 'all'
FRAME_POS_EXTREMES = 'extremes'

YOLO = 'yolov3'
MASKRCNN = 'maskrcnn'