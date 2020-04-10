# cd /media/david/datos/PAPERS-SOURCE_CODE/violencedetection
# cd /Users/davidchoqueluqueroman/Desktop/PAPERS-CODIGOS/violencedetection2
python3 LOCALIZATION/localization_main.py \
--saliencyModelFile SALIENCY/Models/anomaly/mask_model_10_frames_di__epochs-12.tar \
--batchSize 1 \
--numWorkers 1 \
--shuffle false \
--videoSegmentLength 0 \
--videoBlockLength 20 \
--numDynamicImgsPerBlock 1 \
--personDetector yolov3 \
--positionSegment online \
--overlappingBlock 0 \
--overlappingSegment 0 \
--plot false \
--videoName Arrest003 \
--delay 5

# python3 LOCALIZATION/dense_sampling.py \
# --classifierFile ANOMALYCRIME/checkpoints/resnet18_Finetuned-False-_di-2_fusionType-tempMaxPool_num_epochs-20_videoSegmentLength-30_positionSegment-random-FINAL.pth \
# --batchSize 1 \
# --numWorkers 1 \
# --numDiPerVideos 1 \
# --shuffle true \
# --plot true \
# --videoSegmentLength 30 \



# --saliencyModelFile SALIENCY/Models/anomaly/saliency_model_epochs-10.tar SALIENCY/Models/anomaly/mask_model_30_frames_di.tar \