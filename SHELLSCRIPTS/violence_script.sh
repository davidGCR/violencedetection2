
python3  main_violence.py \
--modelType resnet50 \
--numEpochs 30 \
--numWorkers 4 \
--batchSize 8 \
--foldsNumber 5 \
--featureExtract false \
--dataAumentation false \
--joinType maxTempPool \
--videoSegmentLength 30 \
--numDynamicImagesPerVideo 2 \
--typeTrain train \
--positionSegment begin \
--overlaping 0.68