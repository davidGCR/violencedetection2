python3 VIOLENCE_DETECTION/avgAccuracy.py \
--dataset vif \
--rgbModel RGBCNN-dataset=vif_model=resnet50_frameIdx=14_numEpochs=25_featureExtract=False_fold= \
--dynModel VIF-Model-resnet50,segmentLen-30,numDynIms-1,frameSkip-0,epochs-25,splitType-None,fold-