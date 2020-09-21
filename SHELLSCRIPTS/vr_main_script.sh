python3  VIOLENCE_DETECTION/vr_main.py \
--modelType resnet50 \
--dataset hockey \
--numEpochs 3 \
--numWorkers 4 \
--batchSize 4 \
--freezeConvLayers True \
--joinType maxTempPool \
--videoSegmentLength 10 \
--numDynamicImagesPerVideo 1 \
--positionSegment begin \
--overlapping 0 \
--frameSkip 0 \
--saveCheckpoint False \
--splitType openSet-train \
# --useKeyframes blur-max \
# --windowLen 10