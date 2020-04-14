#  python3 plot.py \
# --learningCurvesFolder ANOMALYCRIME/learning_curves \
# --modelName resnet18_Finetuned-True-_di-3_fusionType-maxTempPool_num_epochs-23_videoSegmentLength-30_positionSegment-begin \
# --numFolds 1 \
# --lastEpoch 26
# --modelName resnet18_Finetuned-True-_di-1_fusionType-maxTempPool_num_epochs-30-aumented-data-30 \

 python3 plot.py \
--learningCurve  ANOMALY_RESULTS/learning_curves/trainresnet50-1-Finetuned:True-maxTempPool-numEpochs:10-videoSegmentLength:30-overlaping:0.0-only_violence:True-skipFrame:10 \
--nFolds 1 \
--mode train \
--lastEpoch 4