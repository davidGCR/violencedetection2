#  python3 plot.py \
# --learningCurvesFolder ANOMALYCRIME/learning_curves \
# --modelName resnet18_Finetuned-True-_di-3_fusionType-maxTempPool_num_epochs-23_videoSegmentLength-30_positionSegment-begin \
# --numFolds 1 \
# --lastEpoch 26


 python3 plot.py \
--learningCurvesFolder ANOMALYCRIME/learning_curves \
--modelName resnet18_Finetuned-True-_di-1_fusionType-maxTempPool_num_epochs-30-aumented-data-30 \
--numFolds 1 \
--onlyTrain False \
--lastEpoch 30