python3  VIOLENCE_DETECTION/rgb_cnn.py \
--dataset rwf-2000 \
--model resnet50 \
--numEpochs 25 \
--numWorkers 4 \
--batchSize 8 \
--featureExtract false \
--frameIdx 75 \
--saveCheckpoint false