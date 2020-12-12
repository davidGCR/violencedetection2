import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from VIOLENCE_DETECTION.CAM import compute_CAM, cam2bb
from VIOLENCE_DETECTION.datasetsMemoryLoader import load_fold_data
from VIDEO_REPRESENTATION.dynamicImage import getDynamicImage
from VIOLENCE_DETECTION.metrics import loc_error
from MODELS.ViolenceModels import ResNet_ROI_Pool, ResNet
from VIOLENCE_DETECTION.violenceDataset import ViolenceDataset
import cv2
import torch
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.cluster import KMeans
from collections import Counter
from constants import DEVICE
from VIOLENCE_DETECTION.UTIL2 import base_dataset, transforms_dataset, plot_example
from matplotlib import cm
from PIL import Image
from torchvision import transforms
import constants
import matplotlib.patches as patches
import json

dataset = 'ucfcrime2local'
# if dataset == 'rwf-2000':
#     train_x, train_y, train_numFrames, test_x, test_y, test_numFrames, transforms = base_dataset(dataset)

datasetAll, labelsAll, numFramesAll, transforms_p = base_dataset(dataset, input_size=224)

dataset = ViolenceDataset(videos=datasetAll,
                                labels=labelsAll,
                                numFrames=numFramesAll,
                                spatial_transform=transforms_p['val'],
                                numDynamicImagesPerVideo=5,
                                videoSegmentLength=10,
                                positionSegment='begin',
                                overlaping=0,
                                frame_skip=0,
                                skipInitialFrames=0,
                                ppType=None,
                                useKeyframes=None,
                                windowLen=None,
                                dataset=dataset)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)

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

def denoise(frame, gray):
    if gray:
        frame = cv2.fastNlMeansDenoising(frame,None,10,7,21)
    else:
        frame = cv2.fastNlMeansDenoisingColored(frame,None,10,10,7,21)
    return frame

def palette(clusters):
    width=300
    palette = np.zeros((50, width, 3), np.uint8)
    steps = width/clusters.cluster_centers_.shape[0]
    for idx, centers in enumerate(clusters.cluster_centers_):
        palette[:, int(idx*steps):(int((idx+1)*steps)), :] = centers
    return palette

def palette_perc(k_cluster):
    width = 300
    palette = np.zeros((50, width, 3), np.uint8)

    n_pixels = len(k_cluster.labels_)
    counter = Counter(k_cluster.labels_) # count how many pixels per cluster
    print(counter)
    perc = {}
    for i in counter:
        perc[i] = np.round(counter[i]/n_pixels, 2)
    perc = dict(sorted(perc.items()))

    #for logging purposes
    print(perc)
    print(k_cluster.cluster_centers_)

    step = 0

    for idx, centers in enumerate(k_cluster.cluster_centers_):
        palette[:, step:int(step + perc[idx]*width+1), :] = centers
        step += int(perc[idx]*width+1)

    return palette

def frames_mean(frames_list, gray=False):
    if gray:
        for i in range(len(frames_list)):
            frames_list[i] = cv2.cvtColor(frames_list[i], cv2.COLOR_BGR2GRAY)

    # for i in range(len(frames_list)):
    #     frames_list[i]=denoise(frames_list[i],gray)
    medianFrame = np.median(frames_list, axis=0).astype(dtype=np.uint8)
    th, mask = cv2.threshold(medianFrame, 124, 255, cv2.THRESH_BINARY)

    return medianFrame, mask

from torch.utils.data import Dataset, DataLoader

def get_frame_cap(video_dir, one=-1):
    frames=os.listdir(video_dir)
    frames.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    frames = [os.path.join(video_dir, u) for u in frames]
    for  i,f in enumerate(frames):
        frames[i] = cv2.imread(f)
        frames[i] = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
    if one>-1:
        return frames[one]
    return frames

utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')

class ObjectDetectionPipeline:
    def __init__(self, threshold=0.5, device="cpu", cmap_name="tab10_r"):
        # First we need a Transform object to turn numpy arrays to normalised tensors.
        # We are using an SSD300 model that requires 300x300 images.
        # The normalisation values are standard for pretrained pytorch models.
        self.tfms = transforms.Compose([
            transforms.Resize(300),
            transforms.CenterCrop(300),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.precision = 'fp32'

        # Next we need a model. We're setting it to evaluation mode and sending it to the correct device.
        # We get some speedup from the gpu but not as much as we could.
        # A more efficient way to do this would be to collect frames to a buffer,
        # run them through the network as a batch, then output them one by one
    
        self.model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd').eval().to(device)

        # Stop the network from keeping gradients.
        # It's not required but it gives some speedup / reduces memory use.
        for param in self.model.parameters():
            param.requires_grad = False


        self.device = device
        self.threshold = threshold # Confidence threshold for displaying boxes.
        self.cmap = cm.get_cmap(cmap_name) # colour map
        self.classes_to_labels = utils.get_coco_object_dictionary()


    @staticmethod
    def _crop_img(img):
        """Crop an image or batch of images to square"""
        # print('crop len=',len(img.shape))
        if len(img.shape) == 3:
            y = img.shape[0]
            x = img.shape[1]
        elif len(img.shape) == 4:
            y = img.shape[1]
            x = img.shape[2]
        else:
            
            raise ValueError(f"Image shape: {img.shape} invalid")

        out_size = min((y, x))
        startx = x // 2 - out_size // 2
        starty = y // 2 - out_size // 2

        if len(img.shape) == 3:
            return img[starty:starty+out_size, startx:startx+out_size]
        elif len(img.shape) == 4:
            return img[:, starty:starty+out_size, startx:startx+out_size]

    def _get_rois(self, boxes, labels, uris):
        _, name = os.path.split(uris)
        name = name[:-4]
        my_dict = {}
        my_dict['file'] = name
        # rois = []
        if len(boxes) == 0:
            my_dict['data1'] = [-1,-1,-1,-1]
            # rois.append([-1,-1,-1,-1])
        for idx in range(len(boxes)):
            if labels[idx] == 1:
                left, bot, right, top = boxes[idx]
                x, y, w, h = [val * 300 for val in [left, bot, right - left, top - bot]]
                my_dict['data'+str(idx+1)] = [x, y, w, h]
                # rois.append([x, y, w, h])
            # else:
            #     my_dict['data1'] = [-1,-1,-1,-1]
            #     rois.append([-1,-1,-1,-1])
        return my_dict


    def _plot_boxes(self, output_img, labels, rois, confidences):
        """Plot boxes on an image"""
        if rois[0][0] == -1:
            return None
        fig, ax = plt.subplots(1)
        ax.imshow(output_img)
        for idx in range(len(rois)):
            if True:
                [x, y, w, h] = rois[idx]
                rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                ax.text(x, y, "{} {:.0f}%".format(self.classes_to_labels[labels[idx] - 1], confidences[idx]*100), bbox=dict(facecolor='white', alpha=0.5))
        plt.show()

        return output_img

    def __call__(self, uris):
        """
        Now the call method This takes a raw frame from opencv finds the boxes and draws on it.
        """
        if type(uris) == str:
            img = utils.prepare_input(uris)
            img_tens = utils.prepare_tensor([img], self.precision == 'fp16')

            results = utils.decode_results(self.model(img_tens))
            boxes, labels, confidences = utils.pick_best(results[0], self.threshold)

            output_img = img / 2 + 0.5
            rois = self._get_rois(boxes, labels, uris)
            print(rois)
            self._plot_boxes(output_img, labels, rois[1], confidences)

            return rois

        elif type(uris) == list:
            if len(uris) == 0:
                return None
            
            imgs = [utils.prepare_input(img) for img in uris]
            tens_batch = utils.prepare_tensor(imgs, self.precision == 'fp16')

            detections_batch = self.model(tens_batch)
            results_per_input = utils.decode_results(detections_batch) #results= <class 'list'> 150
            
            best_results_per_input = [utils.pick_best(results, self.threshold) for results in results_per_input]
            rois_l = []
            output_imgs = []
            for image_idx in range(len(best_results_per_input)):
                image = imgs[image_idx] / 2 + 0.5
                output_imgs.append(image)
                boxes, labels, confidences = best_results_per_input[image_idx]
                rois_dict = self._get_rois(boxes, labels, uris[image_idx])
                # print(rois)
                rois_l.append(rois_dict)
                # self._plot_boxes(image,labels,rois[1], confidences)
                
            return rois_l

        else:
            raise TypeError(f"Type {type(img)} not understood")
    
    def rois_save(self, rois_l, out_file):
        with open(out_file, 'w') as fout:
            json.dump(rois_l, fout)
        # with open(out_file, 'w') as filehandle:
        #     filehandle.writelines("{}\t {}\n".format(name, rois) for [name, rois] in rois_l)

    def rois_load(self, file):
        count=0
        with open(file) as fp: 
            for line in fp: 
                count += 1
                row=line.split()
                frame = row[0]
                data = row[1]
                print("Line{}".format(count))
                print("---{}".format(frame))
                print("---{}".format(data))
                if count > 9:
                    return

def object_detector():
    detector = ObjectDetectionPipeline(device=DEVICE, threshold=0.5)
    setdata='val/Fight'
    videos_database = os.listdir(os.path.join(constants.PATH_RWF_2000_FRAMES,setdata))
    videos_database = [os.path.join(constants.PATH_RWF_2000_FRAMES, setdata, v) for v in videos_database]
    
    if not os.path.isdir(os.path.join(constants.PATH_RWF_2000_ROIS, setdata)):
        os.mkdir(os.path.join(constants.PATH_RWF_2000_ROIS, setdata))

    # detector.rois_load("/media/david/datos/Violence DATA/DATASETS/RWF-2000/rois/train/Fight/_2RYnSFPD_U_0.txt")
    

    # output_imgs = detector("/media/david/datos/Violence DATA/DATASETS/RWF-2000/frames/train/Fight/dfsafsghg_325/frame50.jpg")
    # frames_pth = os.listdir(videos_database[0])
    # frames_pth = [os.path.join(videos_database[0],f) for f in frames_pth]
    # rois = detector(frames_pth[:6])
    # detector.save(rois, 'rois.txt')
    # print('output_imgs:', len(rois), rois)

    for i,video in enumerate(videos_database):
        print(i, video)
        frames_pth = os.listdir(video)
        frames_pth.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
        frames_pth = [os.path.join(video,f) for f in frames_pth]
        rois = detector(frames_pth)
        _, v_name = os.path.split(video)
        path_out =os.path.join(constants.PATH_RWF_2000_ROIS,setdata,v_name)
        detector.rois_save(rois, path_out)
    #     # plt.figure(figsize=(10, 10))
    #     output_imgs = detector(get_frame_cap(video))
    #     print('output_imgs:', len(output_imgs))
    #     # plt.imshow(output_imgs[2])
    #     # plt.show()


    # precision = 'fp32'
    # ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_math=precision)
    # utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')
    # ssd_model.to('cuda')
    # ssd_model.eval()
    # video = '/media/david/datos/Violence DATA/DATASETS/RWF-2000/frames/train/Fight/0H2s9UJcNJ0_2'

    # uris=os.listdir(video)
    # uris.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    # uris = [os.path.join(video, u) for u in uris]

    # batchSize = 10
    # rois=[]
    # print('video frames len:', len(uris))
    # for i in range(0,len(uris),batchSize):
    #     if True: #i+batchSize < len(uris):
    #         batch = uris[i:i+batchSize]
    #         print('batch:', i, i+batchSize)
    #         inputs = [utils.prepare_input(uri) for uri in batch]
    #         tensor = utils.prepare_tensor(inputs, precision == 'fp16')
    #         with torch.no_grad():
    #             detections_batch = ssd_model(tensor)
    #         print(len(detections_batch))
    #         results_per_input = utils.decode_results(detections_batch)
    #         best_results_per_input = [utils.pick_best(results, 0.40) for results in results_per_input]
    #         classes_to_labels = utils.get_coco_object_dictionary()

    #         import matplotlib.patches as patches
    #         for image_idx in range(len(best_results_per_input)):
    #             fig, ax = plt.subplots(1)
    #             image = inputs[image_idx] / 2 + 0.5
    #             ax.imshow(image)
    #             bboxes, classes, confidences = best_results_per_input[image_idx]
    #             print('clases:',classes)
    #             for idx in range(len(bboxes)):
    #                 if classes[idx]==1:
    #                     left, bot, right, top = bboxes[idx]
    #                     x, y, w, h = [val * 300 for val in [left, bot, right - left, top - bot]]
    #                     rois.append([x,y,w,h])
    #                     print('ROI: ', x, y, w, h)
    #                     rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
    #                     ax.add_patch(rect)
    #                     ax.text(x, y, "{} {:.0f}%".format(classes_to_labels[classes[idx] - 1], confidences[idx]*100), bbox=dict(facecolor='white', alpha=0.5))
    #         plt.show()
    #         if plt.waitforbuttonpress():
    #             break
    # print(len(rois), rois)

def cluster_segmentation(image, k=3, values=[(0,0,0), (124,124,124), (255,255,255)]):
    # image = frame.copy()
    # Reshaping the image into a 2D array of pixels and 3 color values (RGB)
    pixel_vals = image.reshape((-1,3))

    # Convert to float type
    pixel_vals = np.float32(pixel_vals)
    #the below line of code defines the criteria for the algorithm to stop running,
    #which will happen is 100 iterations are run or the epsilon (which is the required accuracy)
    #becomes 85%
    criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)

    # then perform k-means clustering wit h number of clusters defined as 3
    #also random centres are initally chosed for k-means clustering
    retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # convert data into 8-bit values
    centers = np.uint8(centers)
    # print('image:', image.shape)#(240,320,3)
    # print('centers:', centers.shape,  centers)#(3,3)
    # print('labels:', labels.shape)#(76800, 1) (76800,)

    image_tmp = image.copy()
    image_tmp = image_tmp.reshape((-1,3))
    counter = Counter(labels.flatten())

    counter = {k: v for k, v in sorted(counter.items(), key=lambda item: item[1], reverse=True)}
    # values = [(0,0,0), (124,124,124), (255,255,255)]
    for i, key in enumerate(counter.keys()):
        # print('00000000=====',i,key)
        image_tmp[labels.flatten()==int(key)] = values[i]

    # unique, counts = numpy.unique(a, return_counts=True)
    # print('Counter: ', counter)
    # cl1=np.where(labels.flatten()==0)
    # image_tmp[labels.flatten()==0] = 0
    # image_tmp[labels.flatten()==1] = 0.5
    # image_tmp[labels==3] = 1
    # print('label__:', labels[:10])

    segmented_data = centers[labels.flatten()]
    # print('segmented_data:', segmented_data.shape)#(240,320,3)

    # reshape data into the original image dimensions
    segmented_image = segmented_data.reshape((image.shape))
    image_tmp = image_tmp.reshape((image.shape))

    return segmented_image, image_tmp


if __name__ == '__main__':
    # FFT()
    object_detector()

    # for data in dataloader:
    #     (x, idx, dynamicImages, bboxes), labels = data
    #     # vid_name = datasetAll[idx.item()]
    #     # print(vid_name)
    #     print('---inputs=', x.size(), x.type())
    #     print('---dynamicImages=', len(dynamicImages))
    #
    #     imgs = []
    #     for i in range(len(dynamicImages)):
    #         dyn_img = torch.squeeze(dynamicImages[i]).numpy()
    #         imgs.append(dyn_img)
    #
    #     ########################## DENOISING ##########################
    #     gray=False
    #     imgs_denoised = imgs.copy()
    #     for i in range(len(imgs_denoised)):
    #         imgs_denoised[i]=denoise(imgs_denoised[i],gray)
    #
    #     ########################## CLUSTER SEGMENTATION ##########################
    #
    #     for i in range(len(imgs_denoised)):
    #         segmented_image, image_tmp=cluster_segmentation(imgs_denoised[i],3, [(0,0,0), (255,255,255), (255,255,255)])
    #         imgs_denoised[i]=image_tmp
    #
    #     # threshold=124
    #     sum_maps = np.sum(imgs_denoised, axis=0)
    #     sum_maps_gray = cv2.cvtColor(sum_maps.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    #
    #     val_type=2
    #     dilatation_size=5
    #     if val_type == 0:
    #         dilatation_type = cv2.MORPH_RECT
    #     elif val_type == 1:
    #         dilatation_type = cv2.MORPH_CROSS
    #     elif val_type == 2:
    #         dilatation_type = cv2.MORPH_ELLIPSE
    #     element = cv2.getStructuringElement(dilatation_type, (2*dilatation_size + 1, 2*dilatation_size+1), (dilatation_size, dilatation_size))
    #     sum_maps_gray = cv2.dilate(sum_maps_gray, element)
    #     imgs.append(sum_maps_gray)
    #
    #     # cv2.imshow("contours", sum_maps_gray)
    #     # cv2.waitKey(0)
    #     # plt.imshow(sum_maps)
    #     # plt.show()
    #     im2, contours, hierarchy = cv2.findContours(sum_maps_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #
    #     sm=sum_maps.copy().astype(np.uint8)
    #     # cv2.drawContours(sm, contours, -1, (0,255,0), 3)
    #     # cv2.imshow("contours", sm)
    #     # cv2.waitKey(0)
    #
    #     contours_poly = [None]*len(contours)
    #     boundRect = [None]*len(contours)
    #     for i, c in enumerate(contours):
    #         contours_poly[i] = cv2.approxPolyDP(c, 3, True)
    #         boundRect[i] = cv2.boundingRect(contours_poly[i])
    #
    #     # Draw polygonal contour + bonding rects + circles
    #     color=(0,255,0)
    #     for i in range(len(contours)):
    #         # cv2.drawContours(sum_maps, contours_poly, i, color)
    #         cv2.rectangle(sm, (int(boundRect[i][0]), int(boundRect[i][1])), (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 1)
    #
    #     imgs_denoised.append(sm)
    #
    #     ########################## COLOR_PALETTE ##########################
    #     # clt=KMeans(n_clusters=3)
    #     # for i in range(len(imgs_denoised)):
    #     #     clt_1 = clt.fit(imgs_denoised[i].reshape(-1, 3))
    #     #     # plt.imshow(palette(clt_1))
    #     #     plt.imshow(palette_perc(clt_1))
    #     #     plt.show()
    #
    #     ########################## MEAN ##########################
    #
    #     #Mean
    #     # m_frame, mask = frames_mean(imgs,False)
    #     # imgs.append(m_frame)
    #     # imgs.append(mask)
    #
    #     ########################## PLOT ##########################
    #     fig = plt.figure(figsize=(15., 5.))
    #     grid = ImageGrid(fig, 111,  # similar to subplot(111)
    #                      nrows_ncols=(2, len(imgs)),  # creates 2x2 grid of axes
    #                      axes_pad=0.1,  # pad between axes in inch.
    #                      )
    #
    #     for ax, im in zip(grid, imgs+imgs_denoised):
    #         ax.imshow(im, cmap='gray')
    #     plt.show()
