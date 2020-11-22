python3  VIOLENCE_DETECTION/vr_main.py \
--modelType c3d_v2 \
--inputSize 112 \
--dataset ucfcrime2local \
--lr 0.01 \
--useValSplit False \
--numEpochs 50 \
--numWorkers 4 \
--batchSize 8 \
--freezeConvLayers False \
--joinType maxTempPool \
--videoSegmentLength 10 \
--numDynamicImagesPerVideo 16 \
--positionSegment begin \
--skipInitialFrames 20 \
--overlapping 0.5 \
--frameSkip 0 \
--saveCheckpoint True \
--splitType cross-val \
--patience 10 \
--pretrained MODELS/pretrained/c3d-pretrained.pth 
# --pretrained https://download.openmmlab.com/mmaction/recognition/c3d/c3d_sports1m_pretrain_20201016-dcc47ddc.pth
# --transferModel RESULTS/HOCKEY/checkpoints/DYN_Stream-_dataset=[hockey]_model=resnet50_numEpochs=25_freezeConvLayers=True_numDynamicImages=1_segmentLength=30_frameSkip=0_skipInitialFrames=0_overlap=0.0_joinType=maxTempPool_useKeyframes=None_windowLen=0-fold=1.pt
# --useKeyframes blur-max \
# --windowLen 10
