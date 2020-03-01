#  python3 plot.py \
# --learningCurvesFolder ANOMALYCRIME/learning_curves \
# --modelName resnet18_Finetuned-True-_di-3_fusionType-maxTempPool_num_epochs-23_videoSegmentLength-30_positionSegment-begin \
# --numFolds 1 \
# --lastEpoch 26
# --modelName resnet18_Finetuned-True-_di-1_fusionType-maxTempPool_num_epochs-30-aumented-data-30 \

 python3 plot.py \
--learningCurvesFolder VIOLENCE_RESULTS/learningCurves \
--modelName resnet50_Finetuned-True-_di-3_fusionType-maxTempPool_num_epochs-30 \
--numFolds 5 \
--onlyTrain False \
--lastEpoch 30