cd /media/david/datos/PAPERS-SOURCE_CODE/violencedetection
# python3 LOCALIZATION/localization_main.py \
# --saliencyModelFile SALIENCY/Models/anomaly/saliency_model_epochs-12.tar \
# --batchSize 1 \
# --numWorkers 4 \
# --numDiPerVideos 1 \
# --shuffle false \
# --plot true \
# --videoSegmentLength 20 \
# --personDetector yolov3

python3 LOCALIZATION/dense_sampling.py \
--classifierFile ANOMALYCRIME/checkpoints/resnet18_Finetuned-False-_di-2_fusionType-tempMaxPool_num_epochs-20_videoSegmentLength-30_positionSegment-random-FINAL.pth \
--batchSize 1 \
--numWorkers 1 \
--numDiPerVideos 1 \
--shuffle true \
--plot true \
--videoSegmentLength 30 \