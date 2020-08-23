python3  VIOLENCE_DETECTION/vr_main.py \
--modelType resnet50 \
--dataset rwf-2000 \
--numEpochs 3 \
--numWorkers 4 \
--batchSize 8 \
--featureExtract false \
--joinType maxTempPool \
--videoSegmentLength 30 \
--numDynamicImagesPerVideo 1 \
--positionSegment begin \
--overlapping 0 \
--frameSkip 0 \
--saveCheckpoint True \
--split_type cross-val