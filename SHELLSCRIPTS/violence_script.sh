
python3  main.py \
--dataset hockey \
--modelType resnet18 \
--numEpochs 30 \
--numWorkers 4 \
--batchSize 8 \
--featureExtract true \
--joinType maxTempPool \
--videoSegmentLength 30 \
--numDynamicImagesPerVideo 3 \
--typeTrain final