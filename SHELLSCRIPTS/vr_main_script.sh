python3  VIOLENCE_DETECTION/vr_main.py \
--modelType c3d \
--inputSize 112 \
--dataset rwf-2000 \
--numEpochs 15 \
--numWorkers 4 \
--batchSize 8 \
--freezeConvLayers False \
--joinType maxTempPool \
--videoSegmentLength 20 \
--numDynamicImagesPerVideo 16 \
--positionSegment begin \
--overlapping 0.5 \
--frameSkip 0 \
--saveCheckpoint True \
--splitType cross-val \
--patience 10
# --transferModel RESULTS/HOCKEY/checkpoints/DYN_Stream-_dataset=[hockey]_model=resnet50_numEpochs=25_freezeConvLayers=True_numDynamicImages=1_segmentLength=30_frameSkip=0_skipInitialFrames=0_overlap=0.0_joinType=maxTempPool_useKeyframes=None_windowLen=0-fold=1.pt
# --useKeyframes blur-max \
# --windowLen 10
