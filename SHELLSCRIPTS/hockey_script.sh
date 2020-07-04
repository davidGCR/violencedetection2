
python3  VIOLENCE_DETECTION/hockey_detection.py \
--modelType resnet50 \
--numEpochs 25 \
--numWorkers 4 \
--batchSize 8 \
--featureExtract false \
--joinType maxTempPool \
--videoSegmentLength 10 \
--numDynamicImagesPerVideo 1 \
--positionSegment begin \
--overlapping 0 \
--frameSkip 0 \
--segmentPreprocessing False \
--split_type cross-val

# --transferModel RESULTS/HOCKEY/checkpoints/HOCKEY-Model-alexnetv2,segmentLen-30,numDynIms-1,frameSkip-0,segmentPreprocessing-False,epochs-25,split_type-train-test-1-epoch-16.pth \

# python3  VIOLENCE_DETECTION/hockey_detection.py \
# --modelType alexnetv2 \
# --featureExtract True \
# --joinType maxTempPool \
# --numDynamicImagesPerVideo 1 \
# --split_type fully-conv-1

