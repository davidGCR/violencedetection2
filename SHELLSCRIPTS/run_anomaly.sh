# # TEST : ROC - AUC curve
cd /media/david/datos/PAPERS-SOURCE_CODE/violencedetection 
# cd ..
# python3 ANOMALYCRIME/anomaly_main.py \
# --operation testing \
# --ndis 4 \
# --batchSize 4 \
# --numWorkers 4 \
# --testModelFile ANOMALYCRIME/checkpoints/resnet18_frames_Finetuned-False-_di-4_fusionType-tempMaxPool_num_epochs-17_videoSegmentLength-0_positionSegment-random-FINAL.pth

# TRAIN-VALIDATION-FINAL MODEL
python3 ANOMALYCRIME/anomaly_main.py \
--checkpointPath ANOMALYCRIME/checkpoints \
--operation testing \
--testModelFile ANOMALYCRIME/checkpoints/resnet18_Finetuned-False-_di-2_fusionType-tempMaxPool_num_epochs-20_videoSegmentLength-30_positionSegment-random-FINAL.pth \
--modelType resnet18 \
--joinType tempMaxPool \
--featureExtract true \
--numEpochs 20 \
--ndis 2 \
--maxNumFramesOnVideo 0 \
--videoSegmentLength 30 \
--positionSegment random \
--batchSize 8 \
--numWorkers 4 \
--shuffle true


# # SALIENCY: Train U-NET model
# python3 ANOMALYCRIME/saliencyAnomaly_main.py  \
# --batchSize 8 \
# --numEpochs 10 \
# --numWorkers 4  \
# --saliencyCheckout SALIENCY/Models/anomaly \
# --blackBoxFile ANOMALYCRIME/checkpoints/resnet18_frames_Finetuned-False-_di-1_fusionType-tempMaxPool_num_epochs-50_videoSegmentLength-0_positionSegment-random-FINAL.pth \
# --maxNumFramesOnVideo 0 \
# --videoSegmentLength 16 \
# --positionSegment random


# python3 Saliency/saliencyAnomalyTest.py \
# --saliencyModelFile Saliency/Models/anomaly/saliency_model_epochs-10.tar \
# --batchSize 4 \ 
# --numWorkers 1 \
# --numDiPerVideos 1
