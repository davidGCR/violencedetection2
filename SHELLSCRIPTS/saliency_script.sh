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
--classifier RESULTS/HOCKEY/checkpoints/MaskModel_backnone=resnet50_NDI=1_AreaLoss=8_SmoothLoss=0.5_PreservLoss=0.3_AreaLoss2=0.3_epochs=10-epoch-9.pth \
--modelType alexnetv2 \
--numDiPerVideos 1 \
--segmentLen 30 \
--batchSize 8 \
--numEpochs 10 \
--numWorkers 8  \
--saveCheckpoint True