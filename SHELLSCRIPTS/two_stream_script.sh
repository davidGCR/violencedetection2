python3 VIOLENCE_DETECTION/avgAccuracy.py \
--dataset rwf-2000 \
--rgbModel RGBCNN-dataset=rwf-2000_model=resnet50_frameIdx=15_numEpochs=25_featureExtract=False_fold= \
--dynModel DYN_Stream-_dataset=rwf-2000_model=resnet50_numEpochs=25_featureExtract=False_numDynamicImages=1_segmentLength=30_frameSkip=1_skipInitialFrames=0_overlap=0.0_joinType=maxTempPool-fold=