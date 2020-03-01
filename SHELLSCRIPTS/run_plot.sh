#  python3 plot.py \
# --learningCurvesFolder ANOMALYCRIME/learning_curves \
# --modelName resnet18_Finetuned-True-_di-3_fusionType-maxTempPool_num_epochs-23_videoSegmentLength-30_positionSegment-begin \
# --numFolds 1 \
# --lastEpoch 26
# --modelName resnet18_Finetuned-True-_di-1_fusionType-maxTempPool_num_epochs-30-aumented-data-30 \

 python3 plot.py \
--learningCurvesFolder VIOLENCE_RESULTS/learningCurves \
--modelName Using-2-segmentresnet50-1-Finetuned:True-maxTempPool-segmentLength:30-positionSegment:begin-numEpochs:30-dataAumentation:True \
--numFolds 5 \
--onlyTrain False \
--lastEpoch 27