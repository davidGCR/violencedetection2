import argparse
import createTransforms
import torch
import initializeDataset
import saliencyTester

 
def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saliencyModelFile", type=str)
    parser.add_argument("--batchSize", type=int, default=3)
    # parser.add_argument("--numEpochs", type=int, default=10)
    parser.add_argument("--numWorkers", type=int, default=1)
    parser.add_argument("--numDiPerVideos", type=int, default=5)
    args = parser.parse_args()
    interval_duration = 0
    avgmaxDuration = 0
    numDiPerVideos = args.numDiPerVideos
    input_size = 224
    num_classes = 2
    data_transforms = createTransforms(input_size)
    dataset_source = 'frames'
    debugg_mode = False
    batch_size = args.batchSize
    num_workers = args.numWorkers
    saliency_model_file = args.saliencyModelFile
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    saliency_model_file = args.saliencyModelFile
    saliency_model_config = saliency_model_file
    # saliency_model_file = os.path.join(constants.PATH_SALIENCY_MODELS,saliency_model_file)
    test_x, test_y, dataloader = initializeDataset.get_test_dataloader(batch_size, num_workers, debugg_mode, numDiPerVideos, dataset_source, data_transforms, interval_duration, avgmaxDuration)
    saliencyTester.test(saliency_model_file, num_classes, dataloader, test_x, input_size, saliency_model_config, numDiPerVideos)

__main__()