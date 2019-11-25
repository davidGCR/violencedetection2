# # TEST : ROC - AUC curve
cd /media/david/datos/PAPERS-SOURCE_CODE/MyCode 
# cd ..
# python3 ANOMALYCRIME/anomaly_main.py \
# --operation testing \
# --ndis 4 \
# --batchSize 4 \
# --numWorkers 4 \
# --testModelFile ANOMALYCRIME/checkpoints/resnet18_frames_Finetuned-False-_di-4_fusionType-tempMaxPool_num_epochs-17_videoSegmentLength-0_positionSegment-random-FINAL.pth

# # TRAIN-VALIDATION-FINAL MODEL
# python3 ANOMALYCRIME/anomaly_main.py \
# --checkpointPath ANOMALYCRIME/checkpoints \
# --operation trainingFinal \
# --modelType resnet18 \
# --joinType tempMaxPool \
# --featureExtract true \
# --numEpochs 50 \
# --ndis 1 \
# --batchSize 8 \
# --numWorkers 4 \
# --shuffle true \
# --maxNumFramesOnVideo 40 \
# --videoSegmentLength 0 \
# --positionSegment random

# SALIENCY: Train U-NET model
python3 ANOMALYCRIME/saliencyAnomaly_main.py  \
--batchSize 8 \
--numEpochs 10 \
--numWorkers 4  \
--saliencyCheckout SALIENCY/Models/anomaly \
--blackBoxFile ANOMALYCRIME/checkpoints/resnet18_frames_Finetuned-False-_di-1_fusionType-tempMaxPool_num_epochs-50_videoSegmentLength-0_positionSegment-random-FINAL.pth \
--maxNumFramesOnVideo 0 \
--videoSegmentLength 16 \
--positionSegment random


# python3 Saliency/saliencyAnomalyTest.py \
# --saliencyModelFile Saliency/Models/anomaly/saliency_model_epochs-10.tar \
# --batchSize 4 \ 
# --numWorkers 1 \
# --numDiPerVideos 1
