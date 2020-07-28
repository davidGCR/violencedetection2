python3  VIOLENCE_DETECTION/rgb_cnn.py \
--model resnet50 \
--dataset hockey \
--numEpochs 25 \
--numWorkers 4 \
--batchSize 8 \
--featureExtract false \
--frameIdx 14 \
--saveCheckpoint True