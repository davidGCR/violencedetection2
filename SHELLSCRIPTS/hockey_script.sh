
python3  VIOLENCE_DETECTION/hockey_detection.py \
--modelType resnet50 \
--numEpochs 1 \
--numWorkers 4 \
--batchSize 8 \
--featureExtract false \
--joinType maxTempPool \
--videoSegmentLength 40 \
--numDynamicImagesPerVideo 1 \
--positionSegment begin \
--overlapping 0 \
--frameSkip 0 \
--segmentPreprocessing false \
--split_type cross-val

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

