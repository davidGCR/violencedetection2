# # TEST : ROC - AUC curve
cd /Users/davidchoqueluqueroman/Desktop/PAPERS-CODIGOS/violencedetection2
# cd /media/david/datos/PAPERS-SOURCE_CODE/violencedetection 
# python3 ANOMALYCRIME/anomaly_main.py \
# --operation testing \
# --ndis 3 \
# --batchSize 8 \
# --videoSegmentLength 30 \
# --numWorkers 1 \
# --testModelFile ANOMALYCRIME/checkpoints/resnet18_Finetuned-False-_di-3_fusionType-maxTempPool_num_epochs-23_videoSegmentLength-30_positionSegment-begin-FINAL.pth

# TRAIN-MODEL
python3 ANOMALYCRIME/anomaly_main.py \
--mode train \
--modelType resnet50 \
--joinType maxTempPool \
--featureExtract false \
--numEpochs 10 \
--learningRate 0.001 \
--ndis 1 \
--videoSegmentLength 30 \
--positionSegment begin \
--overlaping 0 \
--batchSize 8 \
--numWorkers 4 \
--shuffle true \
--frame_skip 10
# --typeTrain final \
# --transferModel ninguno



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
