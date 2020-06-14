
from PIL import Image, ImageFile, ImageFilter
import cv2
from skimage.measure import compare_ssim
import imutils

class Preprocessor():
    def __init__(self, pType):
        self.pType = pType
    
    def blur(self, sequence, k):
        seq = []
        for frame in sequence:
            frame = frame.filter(ImageFilter.BoxBlur(k))
            seq.append(frame)
        return seq
    
    def binarize(self, grey):
        ret2, thresh_binary = cv2.threshold(diff1,127,255,cv2.THRESH_BINARY)

    def bakgroundFrameDifference(self, current, previous):
        current = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
        previous = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
       
        diff1 = cv2.absdiff(current, previous)
        # while(1):
        #     cv2.imshow('ok', diff1)
        #     if cv2.waitKey(10) & 0xFF == ord('q'):
        #         break
        # print('previous: ', previous.shape)

        (score, diff2) = compare_ssim(current, previous, full=True)
        # diff2 = (diff2 * 255).astype("uint8")
        
        ret1, thresh_otsu = cv2.threshold(diff1, 0, 255, cv2.THRESH_OTSU)
        # thresh_otsu = None
        ret2, thresh_binary = cv2.threshold(diff1,127,255,cv2.THRESH_BINARY)
        # cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cnts = imutils.grab_contours(cnts)
        # frame_diff = (frame_diff * 255).astype("uint8")
        return diff1, diff2, thresh_binary, thresh_otsu
        # return {'diff1':diff1, 'diff2':diff2, 'thresh_otsu':thresh_otsu, 'thresh_binary':thresh_otsu}
        
