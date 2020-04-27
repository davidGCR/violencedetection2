
python3  main_violence.py \
--modelType resnet50 \
--numEpochs 30 \
--numWorkers 4 \
--batchSize 8 \
--featureExtract false \
--joinType maxTempPool \
--videoSegmentLength 10 \
--numDynamicImagesPerVideo 1 \
--positionSegment begin \
--overlaping 0 \
--frameSkip 1 \
--split_type cross-val

