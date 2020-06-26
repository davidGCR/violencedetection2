
python3  VIOLENCE_DETECTION/vif_detection.py \
--modelType alexnetv2 \
--transferModel RESULTS/HOCKEY/checkpoints/HOCKEY-Model-alexnetv2,segmentLen-30,numDynIms-1,frameSkip-0,segmentPreprocessing-False,epochs-25,split_type-train-test-1-epoch-16.pth \
--numEpochs 25 \
--numWorkers 4 \
--batchSize 8 \
--featureExtract false \
--joinType maxTempPool \
--videoSegmentLength 30 \
--numDynamicImagesPerVideo 1 \
--positionSegment begin \
--overlapping 0 \
--frameSkip 0 \
--segmentPreprocessing false \
--saveCheckpoint false \
--split_type cross-val

# python3  VIOLENCE_DETECTION/vif_detection.py \
# --modelType alexnetv2 \
# --transferModel RESULTS/VIF/checkpoints/VIF-Model-alexnetv2,segmentLen-26,numDynIms-1,frameSkip-0,epochs-25,splitType-None,fold-4.tar-epoch-23.pth \
# --numEpochs 25 \
# --numWorkers 4 \
# --batchSize 1 \
# --featureExtract false \
# --joinType maxTempPool \
# --videoSegmentLength 10 \
# --numDynamicImagesPerVideo 3 \
# --positionSegment begin \
# --overlapping 0 \
# --frameSkip 0 \
# --segmentPreprocessing false \
# --saveCheckpoint false \
# --split_type fully-conv

