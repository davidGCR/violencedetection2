cd /media/david/datos/PAPERS-SOURCE_CODE/violencedetection
python3 LOCALIZATION/localization_main.py \
--saliencyModelFile SALIENCY/Models/anomaly/saliency_model_epochs-10.tar \
--batchSize 1 \
--numWorkers 1 \
--numDiPerVideos 5 \
--shuffle false \
--plot false \
--videoSegmentLength 10 \
--personDetector maskrcnn