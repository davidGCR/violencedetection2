
# python3  VIOLENCE_DETECTION/vif_detection.py \
# --modelType alexnetv2 \
# --numEpochs 25 \
# --numWorkers 4 \
# --batchSize 1 \
# --featureExtract false \
# --joinType maxTempPool \
# --videoSegmentLength 26 \
# --numDynamicImagesPerVideo 1 \
# --positionSegment begin \
# --overlapping 0 \
# --frameSkip 0 \
# --segmentPreprocessing false \
# --saveCheckpoint false \
# --split_type fully-conv

python3  VIOLENCE_DETECTION/vif_detection.py \
--modelType alexnetv2 \
--transferModel RESULTS/VIF/checkpoints/VIF-Model-alexnetv2,segmentLen-26,numDynIms-1,frameSkip-0,epochs-25,splitType-None,fold-4.tar-epoch-23.pth \
--numEpochs 25 \
--numWorkers 4 \
--batchSize 1 \
--featureExtract false \
--joinType maxTempPool \
--videoSegmentLength 26 \
--numDynamicImagesPerVideo 1 \
--positionSegment begin \
--overlapping 0 \
--frameSkip 0 \
--segmentPreprocessing false \
--saveCheckpoint false \
--split_type fully-conv

