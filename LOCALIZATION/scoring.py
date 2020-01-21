import torch
from point import Point
from bounding_box import BoundingBox
import torchvision.transforms as transforms
from dynamicImage import *
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
import torch.nn.functional as f
import cv2


h = 240
w = 320
raw_size = (h, w)
resize2raw_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(raw_size),
        transforms.ToTensor()
])

resize2cnn_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        # transforms.Normalize([0.49237782, 0.49160805, 0.48998737], [0.11053326, 0.11088469, 0.11275752] )
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

normalize_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.49237782, 0.49160805, 0.48998737], [0.11053326, 0.11088469, 0.11275752])
])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def getScore(classifier, tensor_region, bbox):

    with torch.set_grad_enabled(False):
        output_patch = classifier(tensor_region)
        p = torch.nn.functional.softmax(output_patch, dim=1)
        # print('p>: ',p.size())
        # scores.extend(p.cpu().numpy())
        valu_patch, indice_patch = torch.max(output_patch, 1)#0 or 1
        
        p = p.cpu()
        anomaly_score = p[0][1].item()
        normal_score = p[0][0].item()
        
        # bbox.score = indice_patch.item()#label
        print('SOFTMAX SCORE: ', type(p), p.size(), p)
        print('--------> label_p', str(indice_patch), 'valores[0 1]: ', str(output_patch.data))
        return  anomaly_score,  indice_patch
        


def getScoresDirect(tensor_images, saliency_bboxes, classifier, transform):
    for bbox in saliency_bboxes:
        crop_region = tensor_images[:,:, int(bbox.pmin.y):int(bbox.pmax.y), int(bbox.pmin.x):int(bbox.pmax.x)]
        img = crop_region.numpy()[0].transpose(1, 2, 0)
        plt.imshow(img)
        plt.show()


def getScoresFromRegions(video_path, tensor_images, saliency_bboxes, classifier, transform):
    #tensor_images : torch.Size([30,240, 320, 3])
    # tensor_images = tensor_images.permute(0,2,3,1)
    print('SCORING input')
    tensor_images = tensor_images[0]  #30 np arrays of [('frame765.jpg',) tensor([0]) tensor([69]) tensor([80]) tensor([142]) tensor([152])]
    frames = []
    for img in tensor_images:
        img1 = Image.open(os.path.join(video_path, img[0][0])).convert("RGB")
        # img1.show()
        img1 = np.array(img1)# (240 x 320 x3)
        # print('img1 :', img1.shape) 
        frames.append(img1)
    
    imgPIL, img = getDynamicImage(frames)
    # imgPIL.show()

    # plt.imshow(img)
    # plt.title('Img dynamic numpy')
    # plt.show()

    # imgTensor = img
    imgTensor = normalize_transform(imgPIL)
    # imgTensor = transforms.ToTensor()(imgPIL)
    # imgTensor = torch.from_numpy(img).float()

    imgTensor = imgTensor.numpy().transpose(1, 2, 0)
    plt.imshow(imgTensor)
    plt.title('SCORING test')
    plt.show()

    # res = 255*imgTensor
    # res = np.uint8(res)
    # res = cv2.fastNlMeansDenoisingColored(res, None, 10, 10, 7, 21)
    
    # plt.imshow(res)
    # plt.title('SCORING DENOISE test')
    # plt.show()

    # imgTensor = res
    # frames = np.stack(frames, axis=0) #(30, 240, 320, 3)
    # print('frames stacked :', frames.shape)
    crop_regions = []
    for i, bbox in enumerate(saliency_bboxes):
        crop_region = imgTensor[int(bbox.pmin.y):int(bbox.pmax.y), int(bbox.pmin.x):int(bbox.pmax.x),:]
        res = cv2.resize(crop_region, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
        # res=np.uint8(255*res)
        # res = cv2.fastNlMeansDenoisingColored(res, None, 10, 10, 7, 21)
        
        tensor = torch.from_numpy(res).float()  #[h,w,3]
        tensor = tensor.permute(2, 0, 1)
        tensor = tensor.unsqueeze(0)  #[1, c, h, w]
        # print('Tensor: ', tensor.size())
        
        
        tensor = tensor.to(device)
        anomaly_score, label = getScore(classifier, tensor, bbox)
        bbox.score = anomaly_score
        # anomaly_score = 0
        # plt.imshow(crop_region)
        # plt.title('crop_region RESIZED['+str(i)+']-score='+str(anomaly_score)+'- label: '+str(label))
        # plt.show()
       
        

    