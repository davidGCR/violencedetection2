import cv2
import numpy as np
from torch.nn import functional as F
import torch
import random as rng


# params = list(net.parameters())
# weight_softmax = np.squeeze(params[-2].data.cpu().numpy())

# generate class activation mapping for the top1 prediction
def genCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    # print('feature_conv size=', feature_conv.shape)
    # print('weight_softmax size=', weight_softmax[0].shape)
    # print('class_idx=', class_idx)
    output_cam = []
    # for idx in class_idx:
        # cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w))) #2048 * 2048x49
    f_map = feature_conv.reshape((nc, h*w))
    weights = weight_softmax[class_idx]
    cam = np.matmul(weights,f_map)#2048 * 2048x49
    # print('Cam w ({})* fmap({})='.format(weights.shape, f_map.shape, cam.shape))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
        # output_cam.append(cv2.resize(cam_img, size_upsample))
    return cam_img


def compute_CAM(net, x, final_conv, image, plot=False):
    # #input preprecessing
    # x = transforms(img_pil)
    #model preparation
    net.eval()
    params = list(net.parameters())
    weight_softmax = np.squeeze(params[-2].data.cpu().numpy())
    
    # hook the feature extractor
    features_blobs = []
    # final_conv = 'convLayers'

    def hook_feature(module, input, output):
        # print('Hook output=',output.size())
        features_blobs.append(output.data.cpu().numpy())

    net._modules.get(final_conv).register_forward_hook(hook_feature)
    #prediction
    y_pred = net(x)
    
    _, preds = torch.max(y_pred, 1)
    # print('preds=', type(preds), preds)
    h_x = F.softmax(y_pred, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    class_idx = idx.numpy()[0]

    # print('X size=',x.size())
    # print('image size=', image.shape)
    # print('Y and Y size =',y, y.size())
    # print('features_blobs len={}, blob size={}'.format(len(features_blobs),features_blobs[0].shape))
    # print('blob size=',features_blobs[0].shape)
    # print('preds=',preds,preds.size())
    # print('probs, idx sort=',probs, idx)
    # print('h_x=',h_x,h_x.size())
    output_cam = genCAM(features_blobs[0], weight_softmax, class_idx)

    height, width, _ = image.shape
    CAM = cv2.resize(output_cam, (width, height))
    
    heatmap = cv2.applyColorMap(CAM, cv2.COLORMAP_JET)

    if plot:
        # cv2.imshow("Image", frames_rgb[int(len(frames_rgb)/2)])
        # key = cv2.waitKey(0)
        
        # cv2.imshow("Input image", image)
        # key = cv2.waitKey(0)    

        # cv2.imshow("CAM1", CAM)
        # key = cv2.waitKey(0)

        cv2.imshow("CAM2", heatmap)
        key = cv2.waitKey(0)
    
    return preds.item(), CAM, heatmap

def cam2bb(CAM, thr=0.7, plot=False): 
    vmin = np.amin(CAM)
    vmax = np.amax(CAM)
    threshold = 0.65*vmax
    ret, thresh1 = cv2.threshold(CAM, threshold, 255, cv2.THRESH_BINARY) 

    # Find contours
    contours, _ = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Approximate contours to polygons + get bounding rects and circles
    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
    # Draw polygonal contour + bonding rects + circles
    x0, y0, w, h = 0,0,0,0
    areas = []
    for i in range(len(contours)):    
        x0=int(boundRect[i][0])
        y0=int(boundRect[i][1])
        w=boundRect[i][2]
        h=boundRect[i][3]
        areas.append(w*h)
    i_max = np.argmax(np.array(areas))
    x0=int(boundRect[i_max][0])
    y0=int(boundRect[i_max][1])
    w=boundRect[i_max][2]
    h=boundRect[i_max][3]

    if plot:
        # color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        color = (255,255,255)
        cv2.drawContours(thresh1, contours_poly, i_max, color)
        cv2.rectangle(thresh1, (x0, y0),(x0+w, y0+h), color, 2)
        cv2.imshow("thresh1", thresh1)
        key = cv2.waitKey(0)
    return x0, y0, w, h





