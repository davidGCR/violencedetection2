python3  VIOLENCE_DETECTION/crime2local_detection.py \
--modelType alexnetv2 \
--numEpochs 2 \
--numWorkers 4 \
--batchSize 8 \
--featureExtract false \
--joinType maxTempPool \
--videoSegmentLength 20 \
--numDynamicImagesPerVideo 1 \
--positionSegment begin \
--overlapping 0 \
--frameSkip 0 \
--split_type cross-val