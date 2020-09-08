import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from include import root, enviroment
import constants
import numpy as np
from PIL import Image
import cv2
from scipy.signal import argrelextrema
from UTIL.util import sortListByStrNumbers
from VIOLENCE_DETECTION.datasetsMemoryLoader import hockeyLoadData, vifLoadData, rwf_load_data
from VIDEO_REPRESENTATION.dynamicImage import getDynamicImage
from operator import itemgetter
import csv
# Class to hold information about each frame
class ReducedVideo:
    """Class for storing Reduced video
    """
    def __init__(self, path, numSelectedFrames, frames_indexes=None):
        self.path = path
        self.numSelectedFrames = numFrames
        self.frames_indexes = frames_indexes

class Frame:
    """Class for storing frame ref
    """
    def __init__(self, frame, sum_abs_diff):
        self.frame = frame
        self.sum_abs_diff = sum_abs_diff

class Configs:
    # Setting local maxima criteria
    USE_LOCAL_MAXIMA = True
    # Lenght of sliding window taking difference
    # len_window = 5
    # Chunk size of Images to be processed at a time in memory
    # max_frames_in_chunk = 2500
    # Type of smoothening window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman' flat window will produce a moving average smoothing.
    window_type = "hanning"
    

class FrameExtractor():
    """Class for extraction of key frames from video : based on sum of absolute differences in LUV colorspace from given video
    """

    def __init__(self, len_window, max_frames_in_chunk=2500):
        # self.FrameExtractor()
        # Setting local maxima criteria
        self.USE_LOCAL_MAXIMA = True
        # Lenght of sliding window taking difference
        self.len_window = len_window
        # Chunk size of Images to be processed at a time in memory
        self.max_frames_in_chunk = max_frames_in_chunk


    def __calculate_frame_difference(self, frame, curr_frame, prev_frame):
        """Function to calculate the difference between current frame and previous frame
        :param frame: frame from the video
        :type frame: numpy array
        :param curr_frame: current frame from the video in LUV format
        :type curr_frame: numpy array
        :param prev_frame: previous frame from the video in LUV format
        :type prev_frame: numpy array
        :return: difference count and frame if None is empty or undefined else None
        :rtype: tuple
        """

        if curr_frame is not None and prev_frame is not None:
            # Calculating difference between current and previous frame
            diff = cv2.absdiff(curr_frame, prev_frame)
            count = np.sum(diff)
            # print('Count=',count)
            frame = Frame(frame, count)
            return count, frame
        return None

    def __process_frame(self, frame, prev_frame, frame_diffs, frames):
        """Function to calculate the difference between current frame and previous frame
        :param frame: frame from the video
        :type frame: numpy array
        :param prev_frame: previous frame from the video in LUV format
        :type prev_frame: numpy array
        :param frame_diffs: list of frame differences
        :type frame_diffs: list of int
        :param frames: list of frames
        :type frames: list of numpy array
        :return: previous frame and current frame
        :rtype: tuple
        """
        # For GrayScale images
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        curr_frame = grey

        # Calculating the frame difference for previous and current frame
        frame_diff = self.__calculate_frame_difference(frame, curr_frame, prev_frame)
        
        if frame_diff is not None:
            count, frame = frame_diff
            frame_diffs.append(count)
            frames.append(frame)
        prev_frame = curr_frame

        return prev_frame, curr_frame
    

    def __compute_differences_generator__(self, frames_list):
        ret = True
        curr_frame = None
        prev_frame = None

        frame_diffs = []
        frames = []
        # chunk_no = 0
        for i in range(0, len(frames_list)):
            # Calling process frame function to calculate the frame difference and adding the difference
            # in **frame_diffs** list and frame to **frames** list
            frame = frames_list[i]
            prev_frame, curr_frame = self.__process_frame(frame, prev_frame, frame_diffs, frames)
        
        return frames, frame_diffs

    def __get_frames_in_local_maxima__(self, frames, frame_diffs):
        """ Internal function for getting local maxima of key frames
        This functions Returns one single image with strongest change from its vicinity of frames
        ( vicinity defined using window length )
        :param object: base class inheritance
        :type object: class:`Object`
        :param frames: list of frames to do local maxima on
        :type frames: `list of images`
        :param frame_diffs: list of frame difference values
        :type frame_diffs: `list of images`
        """
        extracted_key_frames = []
        diff_array = np.array(frame_diffs)
        # Normalizing the frame differences based on windows parameters
        sm_diff_array = self.__smooth__(diff_array, self.len_window)

        # Get the indexes of those frames which have maximum differences
        frame_indexes = np.asarray(argrelextrema(sm_diff_array, np.greater))[0]
        # print('frame_indexes=',len(frame_indexes), frame_indexes)

        for frame_index in frame_indexes:
            extracted_key_frames.append(frames[frame_index - 1].frame)
        return extracted_key_frames, frame_indexes

    def __smooth__(self, x, window_len, window=Configs.window_type):
        """smooth the data using a window with requested size.
        This method is based on the convolution of a scaled window with the signal.
        The signal is prepared by introducing reflected copies of the signal
        (with the window size) in both ends so that transient parts are minimized
        in the begining and end part of the output signal.
        example:
        import numpy as np
        t = np.linspace(-2,2,0.1)
        x = np.sin(t)+np.random.randn(len(t))*0.1
        y = smooth(x)
        see also:
        numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
        scipy.signal.lfilter
        
        :param x: the frame difference list
        :type x: numpy.ndarray
        :param window_len: the dimension of the smoothing window
        :type window_len: slidding window length
        :param window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman' flat window will produce a moving average smoothing.
        :type window: str
        :return: the smoothed signal
        :rtype: ndarray
        """
        # This function takes
        if x.ndim != 1:
            raise (ValueError, "smooth only accepts 1 dimension arrays.")

        if x.size < window_len:
            print(x.size,window_len)
            raise (ValueError, "Input vector needs to be bigger than window size.")

        if window_len < 3:
            return x

        if not window in ["flat", "hanning", "hamming", "bartlett", "blackman"]:
            raise (
                ValueError,
                "Smoothing Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'",
            )

        # Doing row-wise merging of frame differences wrt window length. frame difference
        # by factor of two and subtracting the frame differences from index == window length in reverse direction
        s = np.r_[2 * x[0] - x[window_len:1:-1], x, 2 * x[-1] - x[-1:-window_len:-1]]

        if window == "flat":  # moving average
            w = np.ones(window_len, "d")
        else:
            w = getattr(np, window)(window_len)
        y = np.convolve(w / w.sum(), s, mode="same")
        return y[window_len - 1: - window_len + 1]

    def __extract_candidate_frames_fromFramesList__(self, frames_list):
        """ Pubic function for this module , Given and input video path
        This functions Returns one list of all candidate key-frames
        :param object: base class inheritance
        :type object: class:`Object`
        :param videopath: inputvideo path
        :type videopath: `str`
        :return: opencv.Image.Image objects
        :rtype: list
        """
        extracted_candidate_key_frames = []
        frames, frame_diffs = self.__compute_differences_generator__(frames_list)
        # print('frame=', len(frames))
        # print('frame_diffs=', len(frame_diffs), frame_diffs)
        extracted_candidate_key_frames_chunk = []
        if self.USE_LOCAL_MAXIMA:
            # Getting the frame with maximum frame difference
            extracted_candidate_key_frames_chunk, frames_indexes = self.__get_frames_in_local_maxima__(frames, frame_diffs)
            extracted_candidate_key_frames.extend(extracted_candidate_key_frames_chunk)

        return extracted_candidate_key_frames, frames_indexes
    
    def __save_frame_to_disk__(self, frame, file_path, file_name, file_ext):
        """saves an in-memory numpy image array on drive.
        
        :param frame: In-memory image. This would have been generated by extract_frames_as_images method
        :type frame: numpy.ndarray, required
        :param file_name: name of the image.
        :type file_name: str, required
        :param file_path: Folder location where files needs to be saved
        :type file_path: str, required
        :param file_ext: File extension indicating the file type for example - '.jpg'
        :type file_ext: str, required         
        :return: None
        """

        file_full_path = os.path.join(file_path, file_name + file_ext)
        # print(file_full_path)
        cv2.imwrite(file_full_path, frame)
    
    def __plot_frames_list__(self, frames_list, waitKey=50):
        i = 0
        while i < len(frames_list):
            cv2.imshow('frame', frames_list[i]) 
            # define q as the exit button 
            if cv2.waitKey(waitKey) & 0xFF == ord('q'): 
                break
            i += 1

    def __save__listDicts_csv__(self, ldicts, csv_columns, csv_file):
        with open(csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in ldicts:
                writer.writerow(data)
    
    def __format_dynamic_image__(self, pilImag):
        imgPIL = pilImag.convert("RGB")
        img = np.array(imgPIL)
        return img

    def __variance_of_laplacian__(self, image):
        # compute the Laplacian of the image and then return the focus
        # measure, which is simply the variance of the Laplacian
	    return cv2.Laplacian(image, cv2.CV_64F).var()
    
    def __load_video_frames__(self, video_path, as_grey=False):
        _, video_name = os.path.split(video_path)
        imgs_paths = os.listdir(video_path)
        imgs_paths = sortListByStrNumbers(imgs_paths)
        frames = []
        for i, path in enumerate(imgs_paths):
            frame_path = os.path.join(video_path, path)
            img = cv2.imread(frame_path)
            if as_grey:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            frames.append(img)
        return frames, imgs_paths

    def __compute_frames_blurring_fromList__(self, frames, plot=False):
        blurrings = []
        for i, frame in enumerate(frames):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fm = self.__variance_of_laplacian__(gray)
            blurrings.append(fm)
            if plot:
                text = "Not Blurry"
                if fm < 50:
                    text = "Blurry"
                cv2.putText(frame, "{}: {:.2f}".format(text, fm), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow("Image", frame)
                key = cv2.waitKey(0)
        return blurrings
    
    def __candidate_frames_blur_based__(self, frames, blurrings, criteria, nelem):
        blurrings = np.array(blurrings)
        selected_frames = []
        indexes = []
        if len(frames) > nelem:
            # if criteria == 'blur-max':
            #     indexes = np.argpartition(blurrings, -nelem)[-nelem:]
            # elif criteria == 'blur-min':
            #     indexes = np.argpartition(blurrings, nelem)[:nelem]
            ss = nelem + int(nelem/2)
            if criteria == 'blur-max':
                # indexes = np.argpartition(blurrings, -ss)[-ss:]
                indexes = np.argsort(blurrings)
                indexes = indexes[::-1][:nelem]
            elif criteria == 'blur-min':
                indexes = np.argsort(blurrings)
            # indexes = np.random.choice(indexes, nelem)
            indexes = np.sort(indexes)
            # avg_blur = np.average(np.average(blurrings))
            selected_frames = []
            # indexes = []
            for i in indexes:
                selected_frames.append(frames[i])
        else:
            indexes = np.arange(len(frames))
            selected_frames = frames
        return selected_frames, indexes
    
    def __avg_dynamic_image__(self, frames, tempWindowLen):
        dimages = []
        indices = [x for x in range(0, len(frames), 1)]
        indices_segments = [indices[x:x + tempWindowLen] for x in range(0, len(indices), tempWindowLen)]

        for i, indices_segment in enumerate(indices_segments): #Generate segments using indices
            segment = np.asarray(frames)[indices_segment].tolist()
            
            imgPIL, img = getDynamicImage(segment)
            dimg = self.__format_dynamic_image__(imgPIL)
            dimg = cv2.cvtColor(dimg, cv2.COLOR_RGB2BGR)
            dimages.append(dimg)
        
        # avg = np.sum(np.array(dimages), axis=0)
        avg=0
        
        return dimages, avg
            

    # def __compute_frames_blurring__(self, video_path, plot=False):
    #     blurrings = []
    #     frames_gray = self.__load_video_frames__(video_path, as_grey=True)
    #     for i, frame in enumerate(frames_gray):
    #         fm = self.__variance_of_laplacian__(frame)
    #         blurrings.append(fm)
    #         if plot:
    #             text = "Not Blurry"
    #             if fm < 50:
    #                 text = "Blurry"
    #             cv2.putText(frame, "{}: {:.2f}".format(text, fm), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    #             cv2.imshow("Image", frame)
    #             key = cv2.waitKey(0)
    #     return blurrings
    
    def __analize_frames_differences__(self, video_path, max_frames=50):
        frames, imgs_paths = self.__load_video_frames__(video_path, as_grey=False)
        
        #Original Dynamic Image
        if len(frames) > max_frames:
            frames2di = frames[0:max_frames]
        else:
            frames2di = frames
        imgPIL, img = getDynamicImage(frames2di)
        imgPIL = imgPIL.convert("RGB")
        img = np.array(imgPIL)
        cv2.imshow("Raw Dynamic Image", img)
        key = cv2.waitKey(0)

        candidate_frames, frames_indexes = self.__extract_candidate_frames_fromFramesList__(frames)

        #Dynamic Image from keyframes
        imgPIL, img = getDynamicImage(candidate_frames)
        imgPIL = imgPIL.convert("RGB")
        img = np.array(imgPIL)
        cv2.imshow("Dynamic Image from Keyframes", img)
        key = cv2.waitKey(0)

        # for i, cframe in enumerate(candidate_frames):
        #     text = imgs_paths[frames_indexes[i]]
        #     # show the image
        #     cv2.putText(cframe, "{}: {:.2f}".format(text, i), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        #     cv2.imshow("Image", cframe)
        #     key = cv2.waitKey(0)
        return frames, candidate_frames, frames_indexes


def main():
    extractor = FrameExtractor(len_window=5)
    # datasetAll = [os.path.join(constants.PATH_HOCKEY_FRAMES_VIOLENCE, '24'),
    #                 os.path.join(constants.PATH_VIF_FRAMES, '1', 'Violence', 'crowd_violence__Man_Utd_vs_Roma_Crowd_Trouble__uncychris__ZGI5vlDMpJA'),
    #                 os.path.join(constants.PATH_RWF_2000_FRAMES, 'train', 'Fight', '_2RYnSFPD_U_0')]
    
    # datasetAll = [os.path.join(constants.PATH_RWF_2000_FRAMES, 'train', 'Fight', '_2RYnSFPD_U_0'),
    #                 os.path.join(constants.PATH_RWF_2000_FRAMES, 'train', 'Fight', '_q5Nwh4Z6ao_1'),
    #                 os.path.join(constants.PATH_RWF_2000_FRAMES, 'train', 'NonFight','i2sLegg2JPA_1')]

    # images_folder = os.path.join(constants.PATH_RWF_2000_FRAMES, 'train', 'Fight','GafFu4IZtIA_0')
    # datasetAll, labelsAll, numFramesAll = hockeyLoadData(shuffle=True)
    # datasetAll, labelsAll, numFramesAll, splitsLen = vifLoadData(constants.PATH_VIF_FRAMES)
    train_names, train_labels, train_num_frames, test_names, test_labels, test_num_frames = rwf_load_data()
    datasetAll = train_names + test_names
    
    bb=[]
    for k, video_path in enumerate(datasetAll):
        print()
        print(video_path)
        frames, imgs_paths = extractor.__load_video_frames__(video_path)
        # print(imgs_paths)

        dyn_image, _ = getDynamicImage(frames)
        dyn_image = extractor.__format_dynamic_image__(dyn_image)
        cv2.imshow("dyn_image", dyn_image)
        key = cv2.waitKey(0)

        # dimages, avg = extractor.__avg_dynamic_image__(frames, 20)
        # for d in dimages:
        #     cv2.imshow("di", d)
        #     key = cv2.waitKey(0)
        # print('Avg', avg.shape)
        # cv2.imshow("dyn_image_avg", avg)
        # key = cv2.waitKey(0)

        # candidate_frames, frames_indexes = extractor.__extract_candidate_frames_fromFramesList__(frames)
        
        # dyn_image_keyframes, _ = getDynamicImage(candidate_frames)
        # dyn_image_keyframes = extractor.__format_dynamic_image__(dyn_image_keyframes)
        # cv2.imshow("dyn_image_keyframes", dyn_image_keyframes)
        # key = cv2.waitKey(0)
        # print('Total/keyframes={}/{}'.format(len(frames), len(candidate_frames)))
        
        blurrings = extractor.__compute_frames_blurring_fromList__(frames, plot=False)
        blurrings = np.array(blurrings)
        print('Blurrings ({})--Max={}, Min={}, Avg={}'.format(len(blurrings),np.amax(blurrings), np.amin(blurrings), np.average(blurrings)))
        
        blurrier_frames_max, indexes_max = extractor.__candidate_frames_blur_based__(frames, blurrings, 'blur-max', 30)
        print('Total/blurs={}/{}'.format(len(frames), len(blurrier_frames_max)))
        print('indexes_max=',indexes_max)

        # print('Total/blurrier_frames={}/{}'.format(len(frames), len(blurrier_frames_max)))
        # print('Blurrier indexes=', type(indexes), indexes)
        # print('Blurrier frames=', blurrings[indexes])
        # bb.append(len(blurrier_frames_max))
        
        dyn_image_blur_max, _ = getDynamicImage(blurrier_frames_max)
        dyn_image_blur_max = extractor.__format_dynamic_image__(dyn_image_blur_max)
        cv2.imshow("dyn_image_blur_max", dyn_image_blur_max)
        key = cv2.waitKey(0)

        # blurrier_frames_min, indexes_min = extractor.__candidate_frames_blur_based__(frames, blurrings, 'blur-min', 30)
        # dyn_image_blur_min, _ = getDynamicImage(blurrier_frames_min)
        # dyn_image_blur_min = extractor.__format_dynamic_image__(dyn_image_blur_min)
        # cv2.imshow("dyn_image_blur_min", dyn_image_blur_min)
        # key = cv2.waitKey(0)
    
    print('Avg blurs keyframes=',np.average(np.array(bb)))
        
        
        

    # fs = []
    # dicts = []
    # for k,video_path in enumerate(datasetAll):
    #     _, video_name = os.path.split(video_path)
    #     imgs_paths = os.listdir(video_path)
    #     imgs_paths = sortListByStrNumbers(imgs_paths)
    #     # print(imgs_paths)
    #     frames = []
    #     for i, path in enumerate(imgs_paths):
    #         frame_path = os.path.join(video_path, path)
    #         # img1 = Image.open(frame_path)
    #         # img1 = img1.convert("RGB")
    #         # img = np.array(img1)
    #         img = cv2.imread(frame_path)
    #         frames.append(img)
        
    #     candidate_frames, frames_indexes = extractor.__extract_candidate_frames_fromFramesList__(frames)
    #     fs.append(len(candidate_frames))
    #     print('{}-No frames/candidates frames={}/{}'.format(k + 1, len(frames), len(candidate_frames)))
    #     dictionary = {
    #         'video_path': video_path,
    #         'video_len': len(frames),
    #         'num_key_frames': len(candidate_frames),
    #         'key_frames': frames_indexes
    #     }
    #     dicts.append(dictionary)

    # print('Average selected={}'.format(np.average(np.array(fs))))
    # extractor.__save__listDicts_csv__(ldicts=dicts, csv_columns=['video_path', 'video_len', 'num_key_frames', 'key_frames'], csv_file='rwf_KeyFrames_10.csv')

if __name__ == "__main__":
    main()