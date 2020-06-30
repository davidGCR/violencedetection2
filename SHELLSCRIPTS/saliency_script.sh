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
--classifier RESULTS/HOCKEY/checkpoints/HOCKEY-Model-alexnetv2,segmentLen-30,numDynIms-1,frameSkip-0,segmentPreprocessing-False,epochs-25,split_type-train-test-1-epoch-16.pth \
--modelType alexnetv2 \
--numDiPerVideos 1 \
--segmentLen 30 \
--batchSize 8 \
--numEpochs 10 \
--numWorkers 8  \
--saveCheckpoint True