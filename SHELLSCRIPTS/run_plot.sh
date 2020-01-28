#  python3 plot.py \
# --learningCurvesFolder ANOMALYCRIME/learning_curves \
# --modelName resnet18_Finetuned-True-_di-3_fusionType-maxTempPool_num_epochs-23_videoSegmentLength-30_positionSegment-begin \
# --numFolds 1 \
# --lastEpoch 26


 python3 plot.py \
--learningCurvesFolder VIOLENCE_RESULTS/learningCurves \
--modelName resnet18-3-maxTempPool-segmentLength:30-positionSegment:begin-numEpochs:30 \
--numFolds 5 \
--lastEpoch 30