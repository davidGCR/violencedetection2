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
# python3 ANOMALYCRIME/anomaly_main.py \
# --mode train \
# --modelType resnet18 \
# --joinType maxTempPool \
# --featureExtract false \
# --numEpochs 10 \
# --learningRate 0.001 \
# --scheduler StepLR \
# --ndis 6 \
# --videoSegmentLength 20 \
# --positionSegment begin \
# --overlaping 0.5 \
# --batchSize 8 \
# --numWorkers 2 \
# --shuffle true \
# --frame_skip 0 \
# --split_type train-test
# --boardFolder numDynImgs3-segmentLen30-skip2-onlyrawvideos-resnet18-Epochs30
# --typeTrain final \
# --transferModel ninguno



# parser.add_argument("--numEpochs",type=int,default=30)
#     parser.add_argument("--batchSize", type=int, default=64)
#     parser.add_argument("--numWorkers", type=int, default=4)
#     parser.add_argument("--ndis", type=int, help="num dyn imgs")
#     parser.add_argument("--videoSegmentLength", type=int, default=0)
#     parser.add_argument("--positionSegment", type=str)
#     parser.add_argument("--shuffle", type=lambda x: (str(x).lower() == 'true'), default=False)
#     parser.add_argument("--frame_skip", type=int)
#     parser.add_argument("--overlaping", type=float)
#     parser.add_argument("--split_type", type=str)

python3 ANOMALYCRIME/anomaly_main.py \
--numEpochs 10 \
--ndis 6 \
--videoSegmentLength 20 \
--positionSegment begin \
--overlaping 0 \
--batchSize 8 \
--numWorkers 2 \
--shuffle true \
--frame_skip 0 \
--split_type train-test

