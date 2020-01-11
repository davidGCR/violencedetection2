# # TEST : ROC - AUC curve
cd /media/david/datos/PAPERS-SOURCE_CODE/violencedetection 
# cd ..
# python3 ANOMALYCRIME/anomaly_main.py \
# --operation testing \
# --ndis 4 \
# --batchSize 4 \
# --numWorkers 4 \
# --testModelFile ANOMALYCRIME/checkpoints/resnet18_frames_Finetuned-False-_di-4_fusionType-tempMaxPool_num_epochs-17_videoSegmentLength-0_positionSegment-random-FINAL.pth

# TRAIN-MODEL
python3 ANOMALYCRIME/anomaly_main.py \
--checkpointPath ANOMALYCRIME/checkpoints \
--operation train \
--modelType resnet18 \
--joinType stdTempPool \
--featureExtract true \
--numEpochs 30 \
--ndis 3 \
--maxNumFramesOnVideo 0 \
--videoSegmentLength 30 \
--positionSegment begin \
--batchSize 8 \
--numWorkers 1 \
--shuffle true



# python3 ANOMALYCRIME/anomaly_main.py \
# --checkpointPath ANOMALYCRIME/checkpoints \
# --operation testing \
# --testModelFile ANOMALYCRIME/checkpoints/resnet18_Finetuned-False-_di-2_fusionType-tempMaxPool_num_epochs-20_videoSegmentLength-30_positionSegment-random-FINAL.pth \
# --modelType resnet18 \
# --joinType tempMaxPool \
# --featureExtract true \
# --numEpochs 20 \
# --ndis 2 \
# --maxNumFramesOnVideo 0 \
# --videoSegmentLength 30 \
# --positionSegment random \
# --batchSize 8 \
# --numWorkers 4 \
# --shuffle true





# python3 Saliency/saliencyAnomalyTest.py \
# --saliencyModelFile Saliency/Models/anomaly/saliency_model_epochs-10.tar \
# --batchSize 4 \ 
# --numWorkers 1 \
# --numDiPerVideos 1
