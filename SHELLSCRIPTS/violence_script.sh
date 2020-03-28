
python3  main_violence.py \
--modelType resnet50 \
--numEpochs 30 \
--numWorkers 4 \
--batchSize 8 \
--foldsNumber 5 \
--featureExtract false \
--dataAumentation false \
--joinType maxTempPool \
--videoSegmentLength 10 \
--numDynamicImagesPerVideo 1 \
--trainMode validationMode \
--positionSegment begin \
--overlaping 0.5