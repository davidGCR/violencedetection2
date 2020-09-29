import cv2
import numpy as np
from torch.nn import functional as F
import torch


# params = list(net.parameters())
# weight_softmax = np.squeeze(params[-2].data.cpu().numpy())

# generate class activation mapping for the top1 prediction
def genCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    print('feature_conv size=', feature_conv.shape)
    print('weight_softmax size=', weight_softmax[0].shape)
    print('class_idx=', class_idx)
    output_cam = []
    # for idx in class_idx:
        # cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w))) #2048 * 2048x49
    f_map = feature_conv.reshape((nc, h*w))
    weights = weight_softmax[class_idx]
    cam = np.matmul(weights,f_map)#2048 * 2048x49
    print('Cam w ({})* fmap({})='.format(weights.shape, f_map.shape, cam.shape))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
        # output_cam.append(cv2.resize(cam_img, size_upsample))
    return cam_img


def compute_CAM(net, x, image):
    # #input preprecessing
    # x = transforms(img_pil)
    #model preparation
    net.eval()
    params = list(net.parameters())
    weight_softmax = np.squeeze(params[-2].data.cpu().numpy())
    
    # hook the feature extractor
    features_blobs = []
    final_conv = 'convLayers'

    def hook_feature(module, input, output):
        print('Hook output=',output.size())
        features_blobs.append(output.data.cpu().numpy())

    net._modules.get(final_conv).register_forward_hook(hook_feature)
    #prediction
    y = net(x)
    _, preds = torch.max(y, 1)
    h_x = F.softmax(y, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    class_idx = idx.numpy()[0]

    print('X size=',x.size())
    print('image size=', image.shape)
    print('Y and Y size =',y, y.size())
    print('features_blobs len={}, blob size={}'.format(len(features_blobs),features_blobs[0].shape))
    print('blob size=',features_blobs[0].shape)
    print('preds=',preds,preds.size())
    print('probs, idx sort=',probs, idx)
    print('h_x=',h_x,h_x.size())
    output_cam = genCAM(features_blobs[0], weight_softmax, class_idx)

    height, width, _ = image.shape
    CAM = cv2.resize(output_cam, (width, height))
    heatmap = cv2.applyColorMap(CAM, cv2.COLORMAP_JET)
    # result = heatmap + image
    # result = image

    cv2.imshow("image", image)
    key = cv2.waitKey(0)

    cv2.imshow("CAM", heatmap*0.3)
    key = cv2.waitKey(0)
    # cv2.imwrite('cam.jpg', result)



