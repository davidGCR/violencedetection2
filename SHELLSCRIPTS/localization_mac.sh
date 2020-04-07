# cd /media/david/datos/PAPERS-SOURCE_CODE/violencedetection
cd /Users/davidchoqueluqueroman/Desktop/PAPERS-CODIGOS/violencedetection2
python3 LOCALIZATION/localization_main.py \
--saliencyModelFile SALIENCY/Models/anomaly/mask_model_10_frames_di__epochs-12.tar \
--batchSize 1 \
--numWorkers 4 \
--numDiPerVideos 1 \
--shuffle false \
--videoSegmentLength 10 \
--personDetector yolov3 \
--positionSegment online \
--overlapping 0.5 \
--plot true \
--videoName Arrest028

# Arrest003
# Arrest006
# Arrest028
# Assault012
# Assault015
# Assault030 Ascensor
# Assault031 Crowd
# Assault033
# Burglary034
# Burglary054
# Burglary059
# Burglary068
# Burglary078
# Burglary097
# Burglary099
# Robbery001
# Robbery003
# Robbery031
# Robbery033
# Robbery041
# Robbery077
# Robbery144
# Stealing027
# Stealing028
# Stealing097
# Stealing102
# Vandalism014
# Vandalism019
# Vandalism029
# Vandalism032
# Vandalism050

# python3 LOCALIZATION/dense_sampling.py \
# --classifierFile ANOMALYCRIME/checkpoints/resnet18_Finetuned-False-_di-2_fusionType-tempMaxPool_num_epochs-20_videoSegmentLength-30_positionSegment-random-FINAL.pth \
# --batchSize 1 \
# --numWorkers 1 \
# --numDiPerVideos 1 \
# --shuffle true \
# --plot true \
# --videoSegmentLength 30 \



# --saliencyModelFile SALIENCY/Models/anomaly/saliency_model_epochs-10.tar SALIENCY/Models/anomaly/mask_model_30_frames_di.tar \