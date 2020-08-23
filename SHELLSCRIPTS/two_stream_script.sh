python3 VIOLENCE_DETECTION/avgAccuracy.py \
--dataset ucfcrime2local \
--rgbModel RGBCNN-dataset=ucfcrime2local_model=resnet50_frameIdx=14_numEpochs=25_featureExtract=False_fold= \
--dynModel DYN_Stream-_dataset=ucfcrime2local_model=resnet50_numEpochs=25_featureExtract=False_numDynamicImages=1_segmentLength=30_frameSkip=0_skipInitialFrames=0_overlap=0.0_joinType=maxTempPool-fold=