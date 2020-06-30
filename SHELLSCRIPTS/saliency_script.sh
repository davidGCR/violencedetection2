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
--classifier RESULTS/HOCKEY/checkpoints/HOCKEY-Model-resnet50,segmentLen-20,numDynIms-1,frameSkip-0,segmentPreprocessing-False,epochs-25,split_type-cross-val,fold-2-epoch-24.pth \
--modelType resnet50 \
--numDiPerVideos 1 \
--batchSize 8 \
--numEpochs 10 \
--numWorkers 8  \
--saveModel True