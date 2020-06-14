
python3  VIOLENCE_DETECTION/hockey_detection.py \
--modelType alexnet \
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
--segmentPreprocessing False \
--split_type train-test-1

# python3  main_violence.py \
# --numEpochs 10 \
# --numDynamicImagesPerVideo 1 \
# --videoSegmentLength 10 \
# --positionSegment begin \
# --overlaping 0 \
# --batchSize 8 \
# --numWorkers 2 \
# --shuffle true \
# --frame_skip 1 \
# --split_type train-test

