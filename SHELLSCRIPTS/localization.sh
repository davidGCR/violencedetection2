cd /media/david/datos/PAPERS-SOURCE_CODE/violencedetection
python3 LOCALIZATION/localization_main.py \
--saliencyModelFile SALIENCY/Models/anomaly/saliency_model_epochs-10.tar \
--batchSize 1 \
--numWorkers 4 \
--numDiPerVideos 1 \
--shuffle false \
--plot true \
--videoSegmentLength 8 \
--personDetector yolov3