import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from VIOLENCE_DETECTION.CAM import compute_CAM, cam2bb
from VIOLENCE_DETECTION.datasetsMemoryLoader import load_fold_data
from VIDEO_REPRESENTATION.dynamicImage import getDynamicImage
from VIOLENCE_DETECTION.metrics import loc_error
from MODELS.ViolenceModels import ResNet_ROI_Pool, ResNet
from VIOLENCE_DETECTION.violenceDataset import ViolenceDataset
import cv2
import torch
import numpy as np
from matplotlib import pyplot as plt

from VIOLENCE_DETECTION.UTIL2 import base_dataset, transforms_dataset, plot_example

def background_model(dynamicImages, iters=10):
    # img_0 = dynamicImages[0]
    # img_0 = torch.squeeze(img_0).numpy()
    masks = []
    backSub = cv2.createBackgroundSubtractorKNN(detectShadows=True)
    for i in range(iters):
        if i<len(dynamicImages):
            img_1 = dynamicImages[i]
        else:
            img_1 = dynamicImages[len(dynamicImages)-1]
        img_1 = torch.squeeze(img_1).numpy()
        img_1 = cv2.fastNlMeansDenoisingColored(img_1,None,10,10,7,21)

        fgMask = backSub.apply(img_1)
        print('---iter ({})'.format(i+1))

        cv2.imshow("img_1", img_1)
        key = cv2.waitKey(0)

        # cv2.imshow("fgMask", fgMask)
        # key = cv2.waitKey(0)

        # gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("gray", gray)
        # key = cv2.waitKey(0)

        # threshold = 0.60*np.amax(gray)
        # ret, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)


        # threshold_inv = np.amin(gray) + 60
        # ret, thresh_inv = cv2.threshold(gray, threshold_inv, 255, cv2.THRESH_BINARY)

        # print('max={}, min={}, mean={}, threshold={}, thresh_inv={}'.format(np.amax(gray), np.amin(gray), np.mean(gray), threshold, threshold_inv))

        # cv2.imshow("thresh", thresh)
        # key = cv2.waitKey(0)

        # cv2.imshow("thresh_inv", thresh_inv)
        # key = cv2.waitKey(0)

        # diff = img_1-img_0
        # cv2.imshow("diff", diff)
        # key = cv2.waitKey(0)

        # img_0 = img_1
    # kernel = np.ones((5, 5), np.uint8)
    # fgMask_d = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)
    fgMask_e = cv2.erode(fgMask,np.ones((3, 3), np.uint8),iterations = 1)
    fgMask_d = cv2.dilate(fgMask_e,np.ones((7, 7), np.uint8),iterations = 1)
    cv2.imshow("fgMask", fgMask)
    cv2.imshow("fgMask_d", fgMask_d)
    key = cv2.waitKey(0)

def __weakly_localization_CAM__():

    ### Load model
    # modelPath = os.path.join(
    #     constants.PATH_RESULTS,
    #     'HOCKEY/checkpoints',
    #     'DYN_Stream-_dataset=[hockey]_model=resnet50_numEpochs=25_freezeConvLayers=True_numDynamicImages=1_segmentLength=30_frameSkip=0_skipInitialFrames=0_overlap=0.0_joinType=maxTempPool_useKeyframes=None_windowLen=0-fold=1.pt'
    #     )
    # dataset='hockey'
    # fold = 1

    # modelPath = os.path.join(
    #     constants.PATH_RESULTS,
    #     'UCFCRIME2LOCAL/checkpoints',
    #     'DYN_Stream-_dataset=[ucfcrime2local]_model=resnet50_numEpochs=25_freezeConvLayers=True_numDynamicImages=1_segmentLength=30_frameSkip=0_skipInitialFrames=0_overlap=0.0_joinType=maxTempPool_useKeyframes=None_windowLen=0-fold=4.pt'
    #     )

    # modelPath_rgb = os.path.join(
    #     constants.PATH_RESULTS,
    #     'UCFCRIME2LOCAL/checkpoints',
    #     'RGBCNN-dataset=ucfcrime2local_model=resnet50_frameIdx=14_numEpochs=25_featureExtract=False_fold=4.pt'
    #     )
    dataset='ucfcrime2local'
    fold = 4

    # stream_type='dyn_img'
    # modelPath = os.path.join(
    #     constants.PATH_RESULTS,
    #     'RWF-2000/checkpoints',
    #     'DYN_Stream-_dataset=[rwf-2000]_model=resnet50_numEpochs=25_freezeConvLayers=True_numDynamicImages=1_segmentLength=30_frameSkip=0_skipInitialFrames=0_overlap=0.0_joinType=maxTempPool_useKeyframes=None_windowLen=0-fold=1.pt'
    #     )
    # dataset='rwf-2000'
    # fold = 1

    # modelPath = os.path.join(
    #     constants.PATH_RESULTS,
    #     'VIF/checkpoints',
    #     'DYN_Stream-_dataset=[vif]_model=resnet50_numEpochs=25_freezeConvLayers=True_numDynamicImages=1_segmentLength=30_frameSkip=0_skipInitialFrames=0_overlap=0.0_joinType=maxTempPool_useKeyframes=None_windowLen=0-fold=1.pt'
    #     )
    # dataset='vif'
    # fold = 1

    if dataset == 'rwf-2000':
        train_x, train_y, train_numFrames, test_x, test_y, test_numFrames, transforms = base_dataset(dataset)
    else:
        datasetAll, labelsAll, numFramesAll, transforms = base_dataset(dataset)
        train_idx, test_idx = load_fold_data(dataset, fold=fold)
        # train_x = list(itemgetter(*train_idx)(datasetAll))
        # train_y = list(itemgetter(*train_idx)(labelsAll))
        # train_numFrames = list(itemgetter(*train_idx)(numFramesAll))
        test_x = list(itemgetter(*test_idx)(datasetAll))
        test_y = list(itemgetter(*test_idx)(labelsAll))
        test_numFrames = list(itemgetter(*test_idx)(numFramesAll))

    # model_, args = load_model(modelPath, stream_type='dyn_img')
    # test_dataset = ViolenceDataset(videos=test_x,
    #                                 labels=test_y,
    #                                 numFrames=test_numFrames,
    #                                 spatial_transform=transforms['val'],
    #                                 positionSegment=None,
    #                                 overlaping=args['overlapping'],
    #                                 frame_skip=args['frameSkip'],
    #                                 skipInitialFrames=args['skipInitialFrames'],
    #                                 ppType=None,
    #                                 useKeyframes=args['useKeyframes'],
    #                                 windowLen=args['windowLen'],
    #                                 numDynamicImagesPerVideo=args['numDynamicImages'], #
    #                                 videoSegmentLength=args['videoSegmentLength'],
    #                                 dataset=dataset
    #                                 )

    model_2 = ResNet_ROI_Pool(

        )
    initialize_model(model_name='resnet50',
                                        num_classes=2,
                                        freezeConvLayers=True,
                                        numDiPerVideos=1,
                                        joinType='maxTempPool',
                                        use_pretrained=True)
    test_dataset = ViolenceDataset(videos=test_x,
                                    labels=test_y,
                                    numFrames=test_numFrames,
                                    spatial_transform=transforms['val'],
                                    numDynamicImagesPerVideo=1,
                                    videoSegmentLength=30,
                                    dataset=dataset
                                    )


    # print(model_2)
    # test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=True, num_workers=4)
    # indices =  [8, 0, 1, 2, 9, 11, 3, 5]  # select your indices here as a list
    # subset = torch.utils.data.Subset(test_dataset, indices)

    ## Analysis
    # test_dataset.numDynamicImagesPerVideo = 20
    # test_dataset.videoSegmentLength = 5
    # test_dataset.overlapping = 0.5
    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = 1, shuffle =True, num_workers=1)

    # print(test_dataset.numDynamicImagesPerVideo, test_dataset.videoSegmentLength)
    # for data in dataloader:
    #     inputs, dynamicImages, labels, v_names, _, paths = data
    #     # print('len(dynamicImages)=',len(dynamicImages))
    #     print(v_names)
    #     print(inputs.size())
    #     background_model(dynamicImages, iters=20)
        # print(len(dynamicImages), type(dynamicImages[0]), dynamicImages[0].size())
        # for i, dyn_img in enumerate(dynamicImages):
        #     dyn_img = torch.squeeze(dyn_img).numpy()
        #     # print('Image-{}='.format(i+1), dyn_img.shape)
        #     cv2.imshow("dyn_img", dyn_img)
        #     key = cv2.waitKey(0)

        #     dst = cv2.fastNlMeansDenoisingColored(dyn_img,None,10,10,7,21)
        #     cv2.imshow("dyn_image denoised", dst)
        #     key = cv2.waitKey(0)

        #     gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        #     cv2.imshow("Gray", gray)
        #     key = cv2.waitKey(0)

        #     heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        #     cv2.imshow("heatmap", heatmap)
        #     key = cv2.waitKey(0)

        #     threshold = 0.60*np.amax(gray)
        #     ret, thresh1 = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        #     cv2.imshow("threshold", thresh1)
        #     key = cv2.waitKey(0)


    # ## Dynamic Image CNN
    y, y_preds = [], []
    for data in dataloader:
        # ipts, dynamicImages, label, vid_name
        inputs, dynamicImages, labels, v_names, gt_bboxes, one_box, paths = data
        print(v_names)
        print('---inputs=', inputs.size(), inputs.type())
        print('---one_box=', one_box.size(), one_box.type())
        print('---gt_bboxes=', len(gt_bboxes), type(gt_bboxes))

        # for g in gt_bboxes:
        #     print(g)

        # y = model_2(inputs, one_box)
        # print('y=',type(y))

    #     dyn_img = torch.squeeze(dynamicImages[0]).numpy()
    #     print('Image1=', dyn_img.shape)

    #     cv2.imshow("dyn_img_1", dyn_img)
    #     key = cv2.waitKey(0)

    #     gray = cv2.cvtColor(dyn_img, cv2.COLOR_BGR2GRAY)
    #     cv2.imshow("Gray", gray)
    #     key = cv2.waitKey(0)

    #     heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    #     cv2.imshow("Image", heatmap)
    #     key = cv2.waitKey(0)

    #     print('v_names=',v_names)
    #     label = labels.item()
    #     if label == 1 and dataset=='ucfcrime2local':
    #         # gt_bboxes, one_box = load_bbox_gt(v_names[0], paths[0])
    #         y.append([label]+one_box)
    #     else:
    #         y.append([label]+[None, None, None, None])

        for i,pt in enumerate(paths[0]):
            print('----one_box=',one_box)
            frame = cv2.imread(pt[0])
            x0, y0, w, h = gt_bboxes[i]
            cv2.rectangle(frame, (x0, y0),(x0+w, y0+h), (0,255,0), 2)
            # x1=one_box[0,0]
            # y1=one_box[0,1]
            # x2=one_box[0,0]+one_box[0,2]
            # y2=one_box[0,1]+one_box[0,3]
            # cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
            cv2.imshow("frame", frame)
            key = cv2.waitKey(0)

    #     y_pred, CAM, heatmap = compute_CAM(model_, inputs, 'convLayers', dyn_img, plot=True)
    #     x0, y0, w, h = cam2bb(CAM, plot=False)
    #     y_preds.append([y_pred, x0, y0, w, h])

    # le = loc_error(y, y_preds)
    # print('Localization error={}'.format(le))

    # # RGB CNN
    # model_rgb, args_rgb = load_model(modelPath_rgb, stream_type='rgb')
    # rgb_transforms = transforms_dataset(dataset, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # rgb_dataset = RGBDataset(dataset=test_x,
    #                             labels=test_y,
    #                             numFrames=test_numFrames,
    #                             spatial_transform=rgb_transforms['test'],
    #                             frame_idx=args_rgb['frameIdx'])
    # rgb_dataloader = torch.utils.data.DataLoader(rgb_dataset, batch_size=1, shuffle=False, num_workers=4)


    # y, y_preds = [], []
    # for data in rgb_dataloader:
    #     v_name, inputs, labels, frame_path = data
    #     print('frame_path=',frame_path)
    #     label = labels.item()
    #     frame=cv2.imread(frame_path[0])
    #     if label == 1 and dataset=='ucfcrime2local':
    #         gt_bboxes, one_box = load_bbox_gt(v_name[0], [frame_path])
    #         y.append([label]+one_box)
    #     else:
    #         y.append([label]+[None, None, None, None])
    #         # net, x, final_conv, image, plot=False
    #     y_pred, CAM, heatmap = compute_CAM(model_rgb, inputs, 'layer4', frame, plot=False)
    #     x0, y0, w, h = cam2bb(CAM, plot=False)
    #     y_preds.append([y_pred, x0, y0, w, h])
    # le = loc_error(y, y_preds)
    # print('Localization error RGB={}'.format(le))

def FastVD(i0, i1):
    img0 = cv2.cvtColor(i0, cv2.COLOR_BGR2GRAY)
    dft0 = cv2.dft(np.float32(img0),flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift0 = np.fft.fftshift(dft0)
    magnitude_spectrum0 = 20*np.log(cv2.magnitude(dft_shift0[:,:,0],dft_shift0[:,:,1]))


    img1 = cv2.cvtColor(i1, cv2.COLOR_BGR2GRAY)
    dft1 = cv2.dft(np.float32(img1),flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift1 = np.fft.fftshift(dft1)
    magnitude_spectrum1 = 20*np.log(cv2.magnitude(dft_shift1[:,:,0],dft_shift1[:,:,1]))


    plt.subplot(231),plt.imshow(i0)
    plt.title('Dynamic Image0'), plt.xticks([]), plt.yticks([])
    plt.subplot(232),plt.imshow(img0, cmap = 'gray')
    plt.title('Input Image0'), plt.xticks([]), plt.yticks([])
    plt.subplot(233),plt.imshow(magnitude_spectrum0, cmap = 'gray')
    plt.title('Magnitude Spectrum0'), plt.xticks([]), plt.yticks([])

    plt.subplot(234),plt.imshow(i1)
    plt.title('Dynamic Image1'), plt.xticks([]), plt.yticks([])
    plt.subplot(235),plt.imshow(img1, cmap = 'gray')
    plt.title('Input Image1'), plt.xticks([]), plt.yticks([])
    plt.subplot(236),plt.imshow(magnitude_spectrum1, cmap = 'gray')
    plt.title('Magnitude Spectrum1'), plt.xticks([]), plt.yticks([])
    plt.show()

    return magnitude_spectrum0, magnitude_spectrum1, dft0, dft1

def FFT():
    dataset = 'ucfcrime2local'
    # if dataset == 'rwf-2000':
    #     train_x, train_y, train_numFrames, test_x, test_y, test_numFrames, transforms = base_dataset(dataset)

    datasetAll, labelsAll, numFramesAll, transforms = base_dataset(dataset, input_size=224)

    dataset = ViolenceDataset(videos=datasetAll,
                                    labels=labelsAll,
                                    numFrames=numFramesAll,
                                    spatial_transform=transforms['val'],
                                    numDynamicImagesPerVideo=6,
                                    videoSegmentLength=5,
                                    positionSegment='begin',
                                    overlaping=0,
                                    frame_skip=0,
                                    skipInitialFrames=0,
                                    ppType=None,
                                    useKeyframes=None,
                                    windowLen=None,
                                    dataset=dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
    y, y_preds = [], []
    for data in dataloader:
        # ipts, dynamicImages, label, vid_name
        # inputs, dynamicImages, labels, v_names, gt_bboxes, one_box, paths = data
        (x, idx, dynamicImages, bboxes), labels = data
        vid_name = datasetAll[idx.item()]
        print(vid_name)
        print('---inputs=', x.size(), x.type())
        print('---dynamicImages=', len(dynamicImages))
        print('---bboxes=', type(bboxes), bboxes.size())

        # i0
        dyn_img0 = torch.squeeze(dynamicImages[0]).numpy()
        #i1
        dyn_img1 = torch.squeeze(dynamicImages[3]).numpy()
        m0, m1, _, _ = FastVD(dyn_img0, dyn_img1)
        rr = np.divide(m1,m0)
        plt.imshow(rr, cmap = 'gray')
        plt.show()

        # for i,pt in enumerate(paths[0]):
        #     print('----one_box=',one_box)
        #     frame = cv2.imread(pt[0])
        # im0='/Users/davidchoqueluqueroman/Desktop/PROJECTS-SOURCE-CODES/violencedetection2/DATASETS/HockeyFightsDATASET/frames/violence/1/frame001.jpg'
        # im1='/Users/davidchoqueluqueroman/Desktop/PROJECTS-SOURCE-CODES/violencedetection2/DATASETS/HockeyFightsDATASET/frames/violence/1/frame002.jpg'

        # frames_paths = os.listdir(vid_name)
        # frames_paths.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
        # im0=os.path.join(vid_name,frames_paths[0])
        # im1=os.path.join(vid_name,frames_paths[int(len(frames_paths))-1])
        # frame0 = cv2.imread(im0)
        # frame1 = cv2.imread(im1)
        # m0, m1, dft0, dft1 = FastVD(frame0, frame1)
        # plt.imshow(m1/m0, cmap = 'gray')
        # plt.show()

def object_detector():
    import torch
    precision = 'fp32'
    ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_math=precision)
    utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')
    ssd_model.to('cuda')
    ssd_model.eval()
    uris = [
        'http://images.cocodataset.org/val2017/000000397133.jpg',
        'http://images.cocodataset.org/val2017/000000037777.jpg',
        'http://images.cocodataset.org/val2017/000000252219.jpg'
    ]
    inputs = [utils.prepare_input(uri) for uri in uris]
    tensor = utils.prepare_tensor(inputs, precision == 'fp16')
    with torch.no_grad():
        detections_batch = ssd_model(tensor)

    results_per_input = utils.decode_results(detections_batch)
    best_results_per_input = [utils.pick_best(results, 0.40) for results in results_per_input]
    classes_to_labels = utils.get_coco_object_dictionary()
    from matplotlib import pyplot as plt
    import matplotlib.patches as patches

    for image_idx in range(len(best_results_per_input)):
        fig, ax = plt.subplots(1)
        # Show original, denormalized image...
        image = inputs[image_idx] / 2 + 0.5
        ax.imshow(image)
        # ...with detections
        bboxes, classes, confidences = best_results_per_input[image_idx]
        for idx in range(len(bboxes)):
            left, bot, right, top = bboxes[idx]
            x, y, w, h = [val * 300 for val in [left, bot, right - left, top - bot]]
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x, y, "{} {:.0f}%".format(classes_to_labels[classes[idx] - 1], confidences[idx]*100), bbox=dict(facecolor='white', alpha=0.5))
    plt.show()

if __name__ == '__main__':
    # FFT()
    object_detector()
