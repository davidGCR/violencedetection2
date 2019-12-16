cd /media/david/datos/PAPERS-SOURCE_CODE/violencedetection 
# python3 SALIENCY/gui_saliency.py \
# --saliencyModelFile SALIENCY/Models/anomaly/saliency_model_epochs-10.tar \
# --batchSize 1 \
# --numWorkers 4 \
# --numDiPerVideos 1 \
# --threshold 0.7 \
# --shuffle false

# # SALIENCY: Train U-NET model
python3 SALIENCY/saliencyTrainer.py  \
--batchSize 4 \
--numEpochs 12 \
--numWorkers 4  \
--numDiPerVideos 1 \
--blackBoxFile ANOMALYCRIME/checkpoints/resnet18_Finetuned-False-_di-2_fusionType-tempMaxPool_num_epochs-20_videoSegmentLength-30_positionSegment-random-FINAL.pth \
--maxNumFramesOnVideo 0 \
--videoSegmentLength 30 \
--positionSegment random