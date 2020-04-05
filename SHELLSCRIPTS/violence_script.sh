
python3  main_violence.py \
--modelType resnet50 \
--numEpochs 5 \
--numWorkers 4 \
--batchSize 8 \
--foldsNumber 5 \
--featureExtract false \
--dataAumentation false \
--joinType maxTempPool \
--videoSegmentLength 20 \
--numDynamicImagesPerVideo 1 \
--trainMode validationMode \
--positionSegment begin \
--overlaping 0 \
--frameSkip 1