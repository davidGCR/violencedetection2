#  python3 plot.py \
# --learningCurvesFolder ANOMALYCRIME/learning_curves \
# --modelName resnet18_Finetuned-True-_di-3_fusionType-maxTempPool_num_epochs-23_videoSegmentLength-30_positionSegment-begin \
# --numFolds 1 \
# --lastEpoch 26
# --modelName resnet18_Finetuned-True-_di-1_fusionType-maxTempPool_num_epochs-30-aumented-data-30 \

 python3 plot.py \
--learningCurve  ANOMALYCRIME/learning_curves/trainresnet50-3-Finetuned:True-maxTempPool-numEpochs:30-videoSegmentLength:40-overlaping:0.5-only_violence:True \
--nFolds 1 \
--mode train \
--lastEpoch 24