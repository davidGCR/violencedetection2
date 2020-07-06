# cd /media/david/datos/PAPERS-SOURCE_CODE/violencedetection 
# python3 SALIENCY/gui_saliency.py \
# --saliencyModelFile SALIENCY/Models/anomaly/saliency_model_epochs-10.tar \
# --batchSize 1 \
# --numWorkers 4 \
# --numDiPerVideos 1 \
# --threshold 0.7 \
# --shuffle false

# # SALIENCY: Train U-NET model
python3 SALIENCY/training.py  \
--classifier RESULTS/UCFCRIME2LOCAL/checkpoints/UCFCRIME2LOCAL-Model-resnet50,trainAllModel-True,TransferModel-steplr,segmentLen-40,numDynIms-1,frameSkip-0,epochs-25,skipInitialFrames-10,split_type-cross-val,fold-1.pt \
--batchSize 8 \
--numEpochs 10 \
--numWorkers 8  \
--saveCheckpoint True