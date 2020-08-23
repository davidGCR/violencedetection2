import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure
import cv2
from PIL import Image
import numpy as np
from sklearn.feature_extraction import image
from patchify import patchify, unpatchify
from scipy.stats import entropy
import seaborn as sb
from math import log2

from VIDEO_REPRESENTATION.dynamicImage import getDynamicImage
from UTIL.util import sortListByStrNumbers
import constants

def load_video_segment(path, init_frame, num_frames):
    frames = os.listdir(path)
    frames = sortListByStrNumbers(frames)
    # print('NumFRames=', len(frames))
    segment = []
    n = 0
    for i in range(len(frames)):
        if len(segment) == num_frames:
            break
        if i >= init_frame:
            img_dir = os.path.join(path, frames[i])
            img1 = Image.open(img_dir)
            img1 = img1.convert("RGB")
            img = np.array(img1)
            segment.append(img)
    central_frame = segment[int(len(segment) / 2)]
    return segment, central_frame

def load_multiple_segments(path, segments_num, segment_len):
    frames = os.listdir(path)
    frames = sortListByStrNumbers(frames)



def my_entropy(events, ets=1e-15):
	return - sum([p * log2(p + ets) for p in events])

def normalize_hist(hist):
    np.sum(hist)

def compute_entropy(fd_gray, entropy_type, threshold=0):
    entropies = np.zeros((fd_gray.shape[0], fd_gray.shape[1]))
    for r in range(fd_gray.shape[0]):
        for c in range(fd_gray.shape[1]):
            if entropy_type == 'scipy':
                entropies[r, c] = entropy(fd_gray[r,c].ravel())
            elif entropy_type=='own':
                entropies[r, c] = my_entropy(fd_gray[r,c].ravel())
            # print(r,c)
            # fd, hog_image_rgb = hog(patches[r, c],
            #                         orientations=n_orientations,
            #                         pixels_per_cell=pixels_per_cell,
            #                         cells_per_block=(1, 1),
            #                         visualize=True,
            #                         multichannel=False,
            #                         block_norm='L2')
            # cell_histograms[r, c] = fd
            # print(cell_histograms[r, c])
    if threshold > 0:
        entropies = entropies > threshold
    return entropies


def main():
    segment_len = 5
    splits = [0,5,10,15,20,30]
    
    video_segments = []

    # segment, central_frame = load_video_segment(os.path.join(constants.PATH_VIF_FRAMES, '1/Violence','fans_violence__Slovak_chauvinism_nationalism_extremism_intolerance_and_all_form'), init_frame=0, num_frames=segment_len)
    segment, central_frame = load_video_segment(os.path.join(constants.PATH_VIF_FRAMES, '1/NonViolence','peaceful_football_crowds__MaximsNewsNetwork_SUDAN_ELECTIONS_YASIR_ARMAN_OMAR_AL_BASHIR_UNM'), init_frame=0, num_frames=segment_len)
    # segment, central_frame = load_video_segment(os.path.join(constants.PATH_HOCKEY_FRAMES_VIOLENCE, '127'), init_frame=0, num_frames=segment_len)
    # segment, central_frame = load_video_segment(os.path.join(constants.PATH_UCFCRIME2LOCAL_FRAMES_VIOLENCE, 'Arrest002-VSplit1'), init_frame=10, num_frames=segment_len)
    print('Segment Len=', len(segment))
    imgPIL, img = getDynamicImage(segment)
    imcv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(imcv, cv2.COLOR_BGR2GRAY)
    print('Image shape=', gray.shape) #(288, 360)
    
    # Calculate gradient RGB
    # sobelx = cv2.Sobel(imcv, cv2.CV_64F, 1, 0, ksize=5)
    # sobely = cv2.Sobel(imcv, cv2.CV_64F, 0, 1, ksize=5)
    # mag, angle = cv2.cartToPolar(sobelx, sobelx, angleInDegrees=True)
    # print('Magnitude: ',mag.shape, '-Angles: ', angle.shape)

    pixels_per_cell = (8,8)
    n_orientations = 128
    cells_per_block = (1,1)
    fd_rgb, hog_image_rgb = hog(imcv, orientations=n_orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, visualize=True, multichannel=True, feature_vector=False)
    fd_gray, hog_image_gray = hog(gray, orientations=n_orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, visualize=True, multichannel=False, feature_vector=False)
    print('**hog_image_rgb: ', hog_image_rgb.shape, '-hog_image_gray: ', hog_image_gray.shape)
    print('****descriptor_rgb: ',fd_rgb.shape, '-descriptor_gray: ', fd_gray.shape)   
    rows = 1
    cols = 3

    
    
    # plt.subplot(rows,cols,1),plt.imshow(imcv,cmap = 'gray')
    # plt.title('Original'), plt.xticks([]), plt.yticks([])
    # plt.subplot(rows,cols,2),plt.imshow(sobelx,cmap = 'gray')
    # plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
    # plt.subplot(rows,cols,3),plt.imshow(sobely,cmap = 'gray')
    # plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
    # plt.subplot(rows,cols,4),plt.imshow(hog_image_rgb,cmap = 'gray')
    # plt.title('SKImg HOG'), plt.xticks([]), plt.yticks([])

    # gray_sobelx8u = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=5)
    # gray_sobely8u = cv2.Sobel(gray, cv2.CV_8U, 0, 1, ksize=5)
    # plt.subplot(rows,cols,5),plt.imshow(gray,cmap = 'gray')
    # plt.title('Original_Gray'), plt.xticks([]), plt.yticks([])
    # plt.subplot(rows,cols,6),plt.imshow(gray_sobelx8u,cmap = 'gray')
    # plt.title('Sobel CV_8U x'), plt.xticks([]), plt.yticks([])
    # plt.subplot(rows,cols,7),plt.imshow(gray_sobely8u,cmap = 'gray')
    # plt.title('Sobel CV_8U y'), plt.xticks([]), plt.yticks([])
     
     # Calculate gradient GRAY
    gray_sobelx64 = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    gray_sobely64 = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    magnitudes_gray, orientations_gray = cv2.cartToPolar(gray_sobelx64, gray_sobely64, angleInDegrees=True)
    print('Magnitude_gray: ', magnitudes_gray.shape, '-Angles_gray: ', orientations_gray.shape)
    print('Angles. Max={}, Min={}'.format(np.amax(orientations_gray), np.amin(orientations_gray)))
    
    plt.subplot(rows,cols,1),plt.imshow(gray,cmap = 'gray')
    plt.title('Original_Gray'), plt.xticks([]), plt.yticks([])
    plt.subplot(rows,cols,2),plt.imshow(central_frame)
    plt.title('Frame'), plt.xticks([]), plt.yticks([])
    # plt.subplot(rows,cols,6),plt.imshow(gray_sobelx64,cmap = 'gray')
    # plt.title('Sobel CV_64F x'), plt.xticks([]), plt.yticks([])
    # plt.subplot(rows,cols,7),plt.imshow(gray_sobely64,cmap = 'gray')
    # plt.title('Sobel CV_64F y'), plt.xticks([]), plt.yticks([])
    plt.subplot(rows,cols,3),plt.imshow(hog_image_gray,cmap = 'gray')
    plt.title('SKImg HOG'), plt.xticks([]), plt.yticks([])
    plt.show() 

    entropies = compute_entropy(fd_gray, entropy_type='own', threshold=0)
    heat_map = sb.heatmap(entropies, annot=False)
    plt.title('Entropy')
    plt.show()

    # heat_map_2 = sb.heatmap(entropies_2, annot=False)
    # plt.title('Entropy_2')
    # plt.show()

    # cells_per_block = (4, 4)
    
    # normalized_hist = np.zeros((patches.shape[0], patches.shape[1],n_orientations))
    # entropies_3 = np.zeros(normalized_hist.shape)
    # print('normalized_hist:', normalized_hist.shape, patches.shape[0] - cells_per_block[0])

    # for r in range(0, patches.shape[0] - cells_per_block[0],1):        
    #     for c in range(0, patches.shape[1] - cells_per_block[1],1):
    #         block = cell_histograms[r:r + cells_per_block[0], c:c + cells_per_block[1]]
    #         print('Block:', block.shape)
    #         normalized_block = block / np.sqrt(np.sum(block ** 2))
    #         print('normalized_block:', normalized_block.shape)
    #         normalized_hist[r, c] = normalized_block
    #         print('Normalized=>',normalized_hist[r, c])
    #         entropies_3[r, c] = my_entropy(normalized_hist[r, c])
            
    # # heat_map_3= sb.heatmap(entropies_3, annot=True)
    # # plt.title('Entropy_3')
    # # plt.show()

if __name__ == "__main__":
    main()