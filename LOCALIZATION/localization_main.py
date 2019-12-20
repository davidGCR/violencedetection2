
import sys
sys.path.insert(1, '/media/david/datos/PAPERS-SOURCE_CODE/violencedetection')
import argparse
import ANOMALYCRIME.transforms_anomaly as transforms_anomaly
import ANOMALYCRIME.anomalyInitializeDataset as anomalyInitializeDataset
# import SALIENCY.saliencyTester as saliencyTester

# from saliencyTester import *
# from SALIENCY.saliencyModel  import SaliencyModel
import SALIENCY.saliencyTester as saliencyTester
import constants
import torch
import os
import tkinter
from PIL import Image, ImageFont, ImageDraw, ImageTk
import numpy as np
import cv2
import glob
from localization_utils import tensor2numpy
import localization_utils
from point import Point
from bounding_box import BoundingBox
import matplotlib.pyplot as plt
from YOLOv3 import yolo_inference
import torchvision.transforms as transforms
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import pandas as pd
import torchvision
import MaskRCNN
from torchvision.utils import make_grid

def maskRCNN():
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

def f(image):
        return np.array(image)

def paste_imgs_on_axes(images, axs):
    ims = []    
    r = axs.shape[0]-1
    for idx in range(len(images)):
        img, title = images[idx]
        imag = axs[r, idx].imshow(f(img))
        
        axs[r, idx].set_title(title)
        ims.append(imag)
    return ims

def pytorch_show(img):
    # img = img / 2 + 0.5 
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')

def myplot(r, c, font_size, grid_static_imgs, img_source, real_frames, real_bboxes, saliency_bboxes, persons_in_segment, persons_segment_filtered, anomalous_regions):
    # create a figure with two subplots
    fig, axes = plt.subplots(r, c, figsize=(25, 15))
    plt.subplots_adjust(hspace=0.5, wspace=0, left = 0.03, right = 0.99)

    for i in range(grid_static_imgs.shape[0]-1):
        for j in range(grid_static_imgs.shape[1]):
            # print(type(grid_static_imgs[i,j]), i,j)
            im1 = axes[i, j].imshow(grid_static_imgs[i,j][0])  
            axes[i, j].set_title(grid_static_imgs[i,j][1])
    
    
    img_source = localization_utils.plotOnlyBBoxOnImage(img_source, saliency_bboxes, constants.PIL_YELLOW)
    # img_source = localization_utils.setLabelInImage(img_source,saliency_bboxes, 'saliency','yellow',10,'left_corner','black' )
    image_anomalous, image_anomalous_final = None, None
    ims = []
    for i in range(len(real_frames)):
    #     # real_frame = real_frames[i].copy()
        images = []
        gt_image = localization_utils.plotOnlyBBoxOnImage(real_frames[i].copy(), real_bboxes[i], constants.PIL_RED)

        img_source = localization_utils.plotOnlyBBoxOnImage(img_source, real_bboxes[i], constants.PIL_RED)
        img_source = localization_utils.setLabelInImage(img_source, real_bboxes[i], 'score', constants.PIL_RED, font_size, 'left_corner', 'white')
        img_source = localization_utils.plotOnlyBBoxOnImage(img_source, saliency_bboxes, constants.PIL_YELLOW)
        for s_box in saliency_bboxes:
            img_source = localization_utils.setLabelInImage(img_source, s_box, 'score' ,font_color=constants.PIL_YELLOW,font_size=font_size,pos_text='right_corner',background_color='black' )
        
        
        images.append((img_source,'source image'))
        
        if len(saliency_bboxes) > 0:
            image_saliency = localization_utils.plotOnlyBBoxOnImage(gt_image.copy(), real_bboxes[i], constants.PIL_RED)
            image_saliency = localization_utils.setLabelInImage(image_saliency, real_bboxes[i], 'score', constants.PIL_RED, font_size, 'left_corner', 'black')
            
            image_saliency = localization_utils.plotOnlyBBoxOnImage(image_saliency, saliency_bboxes, constants.PIL_YELLOW)
            for s_box in saliency_bboxes:
                image_saliency = localization_utils.setLabelInImage(image_saliency,s_box, 'score' ,constants.PIL_YELLOW,font_size,'right_corner','black' )
            images.append((image_saliency,'gt image with bboxes'))
            
            image_persons = localization_utils.plotOnlyBBoxOnImage(image_saliency, persons_in_segment[i], constants.PIL_GREEN)
            # image_persons = localization_utils.setLabelInImage(image_persons,persons_in_segment[i], text='person',font_color='black',font_size=font_size,pos_text='left_corner',background_color='white' )
            images.append((image_persons,'persons'))

            image_persons_filt = localization_utils.plotOnlyBBoxOnImage(real_frames[i].copy(), persons_segment_filtered[i], constants.PIL_MAGENTA)
            # image_persons_filt = localization_utils.setLabelInImage(image_persons_filt, persons_segment_filtered[i], text='person_filter', font_color=constants.PIL_MAGENTA, font_size=font_size, pos_text='left_corner', background_color='white')
            
            images.append((image_persons_filt,'persons close'))
            image_anomalous = localization_utils.plotOnlyBBoxOnImage(real_frames[i].copy(), anomalous_regions, constants.PIL_BLUE)
            # image_anomalous = localization_utils.setLabelInImage(image_anomalous, anomalous_regions, text='anomalous', font_color=constants.PIL_BLUE, font_size=font_size, pos_text='left_corner', background_color='white')

            if len(anomalous_regions) > 0:
                segmentBox = localization_utils.getSegmentBBox(real_bboxes)
                image_anomalous = localization_utils.plotOnlyBBoxOnImage(image_anomalous, segmentBox, constants.PIL_YELLOW,)
                # image_anomalous = localization_utils.setLabelInImage(image_anomalous, anomalous_regions, text='segment', font_color=constants.PIL_YELLOW, font_size=font_size, pos_text='left_corner', background_color='white')
                image_anomalous = localization_utils.plotOnlyBBoxOnImage(image_anomalous, anomalous_regions[0], constants.PIL_MAGENTA)
                # image_anomalous = localization_utils.setLabelInImage(image_anomalous, anomalous_regions, text='anomalous', font_color=constants.PIL_MAGENTA, font_size=font_size, pos_text='left_corner', background_color='white')

            images.append((image_anomalous,'abnormal regions'))                    
        
        
        ims.append(paste_imgs_on_axes(images, axes))
    print('ims: ', len(ims))
    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=100)
    # ani.save('RESULTS/animations/animation.mp4', writer=writer)          
    plt.show()
    # ani.save('RESULTS/animations/animation.gif', writer='imagemagick', fps=30)
    #  
def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saliencyModelFile", type=str)
    parser.add_argument("--batchSize", type=int, default=3)
    # parser.add_argument("--numEpochs", type=int, default=10)
    parser.add_argument("--numWorkers", type=int, default=1)
    parser.add_argument("--numDiPerVideos", type=int, default=5)
    parser.add_argument("--shuffle", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--plot", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--videoSegmentLength", type=int, default=15)
    parser.add_argument("--personDetector", type=str, default=constants.YOLO)

    args = parser.parse_args()
    plot = args.plot
    maxNumFramesOnVideo = 0
    videoSegmentLength = args.videoSegmentLength
    personDetector = args.personDetector
    numDiPerVideos = args.numDiPerVideos
    positionSegment = 'random'
    num_classes = 2 #anomalus or not
    input_size = (224,224)
    transforms_dataset = transforms_anomaly.createTransforms(input_size)
    dataset_source = 'frames'
    batch_size = args.batchSize
    num_workers = args.numWorkers
    saliency_model_file = args.saliencyModelFile
    shuffle = args.shuffle
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    threshold = 0.5

    saliency_model_config = saliency_model_file

    path_dataset = constants.PATH_UCFCRIME2LOCAL_FRAMES_REDUCED
    train_videos_path = os.path.join(constants.PATH_UCFCRIME2LOCAL_README, 'Train_split_AD.txt')
    test_videos_path = os.path.join(constants.PATH_UCFCRIME2LOCAL_README, 'Test_split_AD.txt')
    dataloaders_dict, test_names = anomalyInitializeDataset.initialize_final_only_test_anomaly_dataset(path_dataset, train_videos_path, test_videos_path, batch_size,
                                                        num_workers, numDiPerVideos, transforms_dataset, maxNumFramesOnVideo, videoSegmentLength, positionSegment, shuffle)
    tester = saliencyTester.SaliencyTester(saliency_model_file, num_classes, dataloaders_dict['test'], test_names,
                                        input_size, saliency_model_config, numDiPerVideos, threshold)
    img_size = 416
    weights_path = "YOLOv3/weights/yolov3.weights"
    class_path = "YOLOv3/data/coco.names"
    model_def = "YOLOv3/config/yolov3.cfg"
    conf_thres = 0.8
    nms_thres = 0.4
    yolo_model, classes = yolo_inference.initializeYoloV3(img_size, class_path, model_def, weights_path)
    h = 240
    w = 320
    raw_size = (h, w)
    resize_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(raw_size),
        transforms.ToTensor()
        ])

    scale_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            # transforms.Normalize([0.49237782, 0.49160805, 0.48998737], [0.11053326, 0.11088469, 0.11275752] )
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    crop_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.CenterCrop((224,224)),
            transforms.ToTensor(),
            # transforms.Normalize([0.49237782, 0.49160805, 0.48998737], [0.11053326, 0.11088469, 0.11275752] )
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    data_rows = []
    mask_model = maskRCNN()
    # Set up formatting for the movie files
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    type_set_frames = constants.FRAME_POS_ALL
    first = 0
    mask_rcnn_threshold = 0.3
    # fig, ax = plt.subplots()
    classifierFile = '/media/david/datos/PAPERS-SOURCE_CODE/violencedetection/ANOMALYCRIME/checkpoints/resnet18_Finetuned-False-_di-2_fusionType-tempMaxPool_num_epochs-20_videoSegmentLength-30_positionSegment-random-FINAL.pth'
    classifier = torch.load(classifierFile)
    classifier.eval()
    classifier.inferenceMode()


    for i, data in enumerate(dataloaders_dict['test'], 0):
        print("-" * 150)
        #di_images = [1,ndis,3,224,224]
        dis_images, labels, video_name, bbox_segments = data
        print(video_name, dis_images.size(), len(bbox_segments))
        bbox_segments = np.array(bbox_segments)
        ######################## dynamic images
        l_source_frames = []
        l_di_images = [] # to plot
        dis_images = dis_images.detach().cpu() # torch.Size([1, 3, 224, 224])
        
        # dis_images = torch.squeeze(dis_images, 0) ## to num dynamic images > 1 and minibatch == 1
        for di_image in dis_images:
            # di_image = di_image / 2 + 0.5  q    
            di_image = resize_transform(di_image)
            di_image = di_image.numpy()
            di_image = np.transpose(di_image, (1, 2, 0))
            l_di_images.append(di_image)
            # print('di_image: ', di_image.shape)

        ######################## mask
        masks = tester.compute_mask(dis_images, labels)
        masks = torch.squeeze(masks, 0)  #tensor [ndis,1,224,224]
        masks = masks.detach().cpu()
        masks = tester.min_max_normalize_tensor(masks) #to 0-1
        l_masks = []
        
        for mask in masks:
            mask = resize_transform(mask.cpu())
            mask = tensor2numpy(mask)
            l_masks.append(mask)
            # saliency_bboxes, preprocesing_reults = localization_utils.computeBoundingBoxFromMask(masks)  #only one by segment
            

        ######################## dynamic images masked
        dis_masked = dis_images * masks  #tensor [1,3,224,224]
        dis_masked = torch.squeeze(dis_masked, 0)  #tensor [3,224,224]
        # dis_masked = dis_masked.detach().cpu()

        source_frames = masks

        video_prepoc_saliencies = []
        video_prepoc_saliencies2 = []
        l_imgs_processing = []

        for idx_dyn_img, source_frame in enumerate(source_frames): 
            source_frame = resize_transform(source_frame)
            source_frame = tensor2numpy(source_frame)
            
            saliency_bboxes, preprocesing_reults = localization_utils.computeBoundingBoxFromMask(source_frame)  #only one by segment
            if len(saliency_bboxes) > 1:
                # print('****removing inside areas...', len(saliency_bboxes))
                saliency_bboxes = localization_utils.removeInsideSmallAreas(saliency_bboxes)
                # print('****removing inside areas finished...', len(saliency_bboxes))
            
            di_image = resize_transform(dis_images[idx_dyn_img])
            # batch_crop_regions = []
            for bbox in saliency_bboxes:
                # print('Crop region BBOX: ',bbox)
                crop_image = di_image[:, bbox.pmin.y:bbox.pmax.y,bbox.pmin.x:bbox.pmax.x]
                # batch_crop_regions.append(crop_image)
                # batch_crop_regions = torch.stack(batch_crop_regions,dim=0)
                # print('Crop region size: ', crop_image.size())
                if crop_image.size()[1] > 224 or crop_image.size()[2] > 224:
                    crop_image = crop_transform(crop_image)
                else:
                    crop_image = scale_transform(crop_image)
                batch_crop_regions = torch.unsqueeze(crop_image, dim=0)
                batch_crop_regions = batch_crop_regions.to(device)
                # print('batch_crop_regions: ', batch_crop_regions.size())
                with torch.set_grad_enabled(False):
                    output_patch = classifier(batch_crop_regions)
                    p = torch.nn.functional.softmax(output_patch, dim=1)
                    # print('p>: ',p.size())
                    # scores.extend(p.cpu().numpy())
                    valu_patch, indice_patch = torch.max(output_patch, 1)
                    
                    bbox.score = p[0][1].cpu().item()
                    print('--------> label_p' , str(indice_patch),', score: ', str(p.data), bbox)

            #     plot = True
            # else:
            #     plot = False
            # crops = []
            # for bbox in saliency_bboxes:
            #     crop_image = dis_images[:,:, bbox.pmin.x:bbox.pmax.x, bbox.pmin.y:bbox.pmax.y]
            #     # crops.append(crop_image)
            #     crop_image = crop_image / 2 + 0.5
            #     pytorch_show(make_grid(crop_image.cpu().data, nrow=7, padding=10))
            #     plt.show()
            # crops = torch.stack(crops, dim=0)
            # print('crops: ', crops.size())
            
            # 
            
            print('saliency_bboxes: ', type(saliency_bboxes),len(saliency_bboxes),saliency_bboxes[0])
            video_prepoc_saliencies.append({
                'saliency bboxes': saliency_bboxes,
                'preprocesing': preprocesing_reults
            })


            img_thresholding = localization_utils.myTresholding(source_frame, cell_size=[4, 4])
            saliency_bboxes2, preprocesing_reults2 = localization_utils.myPreprocessing(img_thresholding)  #only one by segment
            l_imgs_processing.append(img_thresholding)
            video_prepoc_saliencies2.append({
                'saliency bboxes': saliency_bboxes2,
                'preprocesing': preprocesing_reults2
            })
            

            source_frame = localization_utils.gray2rgbRepeat(np.squeeze(source_frame,2))
            l_source_frames.append(source_frame)
            saliency_bboxes, preprocesing_reults = localization_utils.computeBoundingBoxFromMask(source_frame)  #only one by segment
            
            
        
        video_real_info = []
        
        for bbox_segment in bbox_segments:
            # print('bbox_segment: ',bbox_segment)
            #read frames of segment
            frames_names, real_frames, real_bboxes = localization_utils.getFramesFromSegment(video_name[0], bbox_segment, 'all')
            video_real_info.append({
                'frames_names':frames_names,
                'real_frames':real_frames,
                'real_bboxes': real_bboxes})
          
            # img_test = tt(real_frames[0])
            # img_test = torch.unsqueeze(img_test,dim=0)
            
        # img_test = img_test * masks  #tensor [1,3,224,224]
        # pytorch_show(make_grid(img_test.data, padding=10))
        
        for index, segment_real_info in enumerate(video_real_info): #di by di
            #person detection
            persons_in_segment = []
            anomalous_regions = []  # to plot
            persons_in_frame = []
            persons_segment_filtered = [] #only to vizualice
            for idx, frame in enumerate(segment_real_info['real_frames']):
                if personDetector == constants.YOLO:
                    persons_in_frame = localization_utils.personDetectionInFrame(yolo_model, img_size, conf_thres, nms_thres, classes, frame)
                    # print('persons_in_frame MaskRCNN: ', len(persons_in_frame))
                elif personDetector == constants.MASKRCNN:
                    persons_in_frame = MaskRCNN.personDetectionInFrameMaskRCNN(mask_model, frame, mask_rcnn_threshold)
                # print('num Persons in frame: ', len(persons_in_frame))
                persons_in_segment.append(persons_in_frame)
        
            #refinement in segment
            for i,persons_in_frame in enumerate(persons_in_segment):
                persons_filtered, only_joined_regions = localization_utils.filterClosePersonsInFrame(persons_in_frame, 20)
                persons_segment_filtered.append(only_joined_regions)
                # print('persons_in_frame filter: ', len(persons_filtered))
                anomalous_regions_in_frame = localization_utils.findAnomalyRegionsOnFrame(persons_filtered, video_prepoc_saliencies[index]['saliency bboxes'], 0.3)
                # anomalous_regions_in_frame = sorted(anomalous_regions_in_frame, key=lambda x: x.iou, reverse=True)
                
                anomalous_regions.extend(anomalous_regions_in_frame)
                # print('--------------- anomalous_regions: ', len(anomalous_regions))
                anomalous_regions.sort(key=lambda x: x.iou, reverse=True) #sort by iuo to get only one anomalous regions
            ######################################              
            segmentBox = localization_utils.getSegmentBBox(segment_real_info['real_bboxes'])
            # di_image = resize_transform(dis_images[idx_dyn_img])
            di_image = resize_transform(dis_images[0])
            # print('fdsfgsdgdsfgfdgh: ', di_image.size())
            crop_image = di_image[:, int(segmentBox.pmin.y):int(segmentBox.pmax.y), int(segmentBox.pmin.x):int(segmentBox.pmax.x)]
            if crop_image.size()[1] > 224 or crop_image.size()[2] > 224:
                crop_image = crop_transform(crop_image)
            else:
                crop_image = scale_transform(crop_image)
            batch_crop_regions = torch.unsqueeze(crop_image, dim=0)
            batch_crop_regions = batch_crop_regions.to(device)
            with torch.set_grad_enabled(False):
                output_patch = classifier(batch_crop_regions)
                p = torch.nn.functional.softmax(output_patch, dim=1)
                valu_patch, indice_patch = torch.max(output_patch, 1)

                for b in segment_real_info['real_bboxes']:
                    b.score = p[0][1].cpu().item()
                
                print('--------> label_real' , str(indice_patch),', score: ', str(p.data), segmentBox)
            ######################################
            if len(anomalous_regions) == 0:
                # print('No anomalous regions found...')
                iou = localization_utils.IOU(segment_real_info['real_bboxes'][0],None)
            else:
                # segmentBox = localization_utils.getSegmentBBox(segment_real_info['real_bboxes'])
                iou = localization_utils.IOU(segmentBox,anomalous_regions[0])
            row = [frames_names[first], iou]
            data_rows.append(row)
        
            if plot:
                font_size = 9
                subplot_r = 2
                subplot_c = 5
                img_source = l_source_frames[index]
                di_image = l_di_images[index]
                di_image = di_image / 2 + 0.5

                
                
                # preprocesing_reults = video_prepoc_saliencies[index]['preprocesing']
                preprocesing_reults = np.empty( (subplot_r,subplot_c), dtype=tuple)
                preprocesing_reults[0,0] = (img_source,'image source')
                # mask = l_masks[index]
                # mask = np.squeeze(mask,2)
                # mask = np.stack([mask, mask, mask], axis=2)
                # preprocesing_reults.append((mask,'mask'))
                # preprocesing_reults.append((img_source,'img source'))
                preprocesing_reults[0,1]=(video_prepoc_saliencies[index]['preprocesing'][0], 'thresholding')
                preprocesing_reults[0,2]=(video_prepoc_saliencies[index]['preprocesing'][1], 'morpho')
                preprocesing_reults[0,3]=(video_prepoc_saliencies[index]['preprocesing'][2], 'contours')
                preprocesing_reults[0,4]=(video_prepoc_saliencies[index]['preprocesing'][3], 'bboxes')

                # img_myThresholding = l_imgs_processing[index]
                # img_myThresholding = np.squeeze(img_myThresholding,2)
                # img_myThresholding = np.stack([img_myThresholding, img_myThresholding, img_myThresholding], axis=2)
                # preprocesing_reults[1,0]=(img_myThresholding, 'myThresholding')
                # preprocesing_reults[1,1]=(video_prepoc_saliencies2[index]['preprocesing'][0], 'thresholding2')
                # preprocesing_reults[1,2]=(video_prepoc_saliencies2[index]['preprocesing'][1], 'morpho2')
                # preprocesing_reults[1,3]=(video_prepoc_saliencies2[index]['preprocesing'][2], 'contours2')
                # preprocesing_reults[1,4]=(video_prepoc_saliencies2[index]['preprocesing'][3], 'bboxes2')
               

                real_frames = video_real_info[index]['real_frames']
                real_bboxes = video_real_info[index]['real_bboxes']

                # preprocesing_reults.append((np.multiply(real_frames[0],mask),'realFrame x mask'))
                
                saliency_bboxes = video_prepoc_saliencies[index]['saliency bboxes']
                
                # print('types: ', type(dynamic_img), type(real_frames), type(real_bboxes))
                myplot(subplot_r, subplot_c, font_size, preprocesing_reults, di_image, real_frames, real_bboxes, saliency_bboxes, persons_in_segment, persons_segment_filtered, anomalous_regions)
                
                
                # di_image_to_crop = resize_transform(torch.squeeze(dis_images.cpu(), dim=0))
                # di_image_to_crop = torch.unsqueeze(di_image_to_crop, dim=0)
                # for bbox in saliency_bboxes:
                #     print('Bounding Box: ', bbox.pmin.x, bbox.pmin.y,bbox.pmax.x,bbox.pmax.y)
                #     crop_image = di_image_to_crop[:,:,  bbox.pmin.y:bbox.pmax.y ,bbox.pmin.x:bbox.pmax.x]
                #     # crops.append(crop_image)
                #     crop_image = crop_image / 2 + 0.5
                #     pytorch_show(make_grid(crop_image.cpu().data, nrow=7, padding=10))
                #     plt.show()
    
    # ############# MAP #################
    print('data rows: ', len(data_rows))
    df = pd.DataFrame(data_rows, columns=['path', 'iou'])
    df['tp/fp'] = df['iou'].apply(lambda x: 'TP' if x >= 0.5 else 'FP')
    localization_utils.mAP(df)
   
__main__()


