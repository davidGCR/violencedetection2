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
    
    def __save_frame_to_disk__(self, frame, file_path):
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

        file_full_path = os.path.join(file_path)
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

    def __load_frames_from_list__(self, video_path, imgs_paths, as_grey=False):
        frames = []
        for i, path in enumerate(imgs_paths):
            frame_path = os.path.join(video_path, path)
            img = cv2.imread(frame_path)
            if as_grey:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            frames.append(img)
        return frames

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
        # selected_frames = []
        indexes = []
        if len(frames) > nelem:
            # if criteria == 'blur-max':
            #     indexes = np.argpartition(blurrings, -nelem)[-nelem:]
            # elif criteria == 'blur-min':
            #     indexes = np.argpartition(blurrings, nelem)[:nelem]
            # ss = nelem + int(nelem/2)
            if criteria == 'blur-max':
                # indexes = np.argpartition(blurrings, -ss)[-ss:]
                indexes = np.argsort(blurrings)
                indexes = indexes[::-1][:nelem]
            elif criteria == 'blur-min':
                indexes = np.argsort(blurrings)
                indexes = indexes[:nelem]
            # indexes = np.random.choice(indexes, nelem)
            # indexes = np.sort(indexes)
            # avg_blur = np.average(np.average(blurrings))
            # selected_frames = []
            # indexes = []
            # for i in indexes:
            #     selected_frames.append(frames[i])
        else:
            indexes = np.arange(len(frames))
            # selected_frames = frames
        return indexes.tolist()
    
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

    def __padding__(self, segment_list, numDynImgs):
        if  len(segment_list) < numDynImgs:
            last_element = segment_list[len(segment_list) - 1]
            for i in range(numDynImgs - len(segment_list)):
                segment_list.append(last_element)
        elif len(segment_list) > numDynImgs:
            segment_list = segment_list[0:numDynImgs]
        return segment_list

    def __checkSegmentLength__(self, segment, seqLen, minSegmentLen=4):
        return (len(segment) == seqLen or len(segment) > minSegmentLen)

    def __getVideoSegments__(self, vid_path, numDynImgs=1, seqLen=0, overlaping=0):
        frames_list = os.listdir(vid_path)
        frames_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
        # video_segments = []
        indices = [x for x in range(0, len(frames_list), 1)]
        
        if seqLen == 0:
            seqLen = len(frames_list)
        # print('seqLen = ', seqLen)
        overlap_length = int(overlaping*seqLen)
        indices_segments = [indices[x:x + seqLen] for x in range(0, len(indices), seqLen-overlap_length)]
        
        indices_segments_cpy = []
        for i,seg in enumerate(indices_segments): #Verify is a segment has at least 2 or more frames.
            if self.__checkSegmentLength__(seg, seqLen):
                indices_segments_cpy.append(indices_segments[i])
        indices_segments = indices_segments_cpy
        
        # indices_segments = self.__padding__(indices_segments, numDynImgs) #If numDynamicImages < wanted the padding else delete
        # print('indices3: ', len(indices_segments), indices_segments)

        # for i, indices_segment in enumerate(indices_segments): #Generate segments using indices
        #     segment = np.asarray(frames_list)[indices_segment].tolist()
        #     video_segments.append(segment)
                
        return indices_segments

def vif_analysis():
    from VIOLENCE_DETECTION.datasetsMemoryLoader import customize_kfold

    datasetAll, labelsAll, numFramesAll, splitsLen = vifLoadData(constants.PATH_VIF_FRAMES)

    # for train_idx, test_idx in customize_kfold(n_splits=folds_number, dataset=args.dataset, X_len=len(datasetAll), shuffle=shuffle):

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
    
    bb = []
    nelem = 1
    seqLen = 15
    dynamicImgLen = [5, 10, 15, 20]
    for k, video_path in enumerate(datasetAll):
        print()
        
        _, video_name = os.path.split(video_path)

        frames, imgs_paths = extractor.__load_video_frames__(video_path) # Load all frames
        print(len(imgs_paths), video_path)
        indices_segments = extractor.__getVideoSegments__(video_path, seqLen=seqLen) # split all video in shots of seqLen frames
        # print(len(video_segments), video_segments)
        print(indices_segments)
        segment_idxs = []
        for i,idxs in enumerate(indices_segments):

            frames_in_segment = list(itemgetter(*idxs)(frames)) #np.asarray(frames)[idxs].tolist()
            blurrings = extractor.__compute_frames_blurring_fromList__(frames_in_segment, plot=False)
            # print('blurrings=',blurrings)
            indexes_candidates = extractor.__candidate_frames_blur_based__(frames_in_segment, blurrings, 'blur-min', nelem=nelem)
            # print('indexes_candidates1=',indexes_candidates, type(indexes_candidates))

            if nelem > 1:
                indexes_candidates = list(itemgetter(*indexes_candidates)(idxs))
                segment_idxs += indexes_candidates
                print('Segment i{}-(len={}/candidates={})'.format(i+1,len(frames_in_segment), len(indexes_candidates)))
            else:
                indexes_candidates = idxs[indexes_candidates[0]]
                segment_idxs.append(indexes_candidates)
                print('Segment i{}-(len={}/candidates={})'.format(i+1,len(frames_in_segment), 1))
            
            print('indexes_candidates=',indexes_candidates)
            

        segment_idxs.sort()
        candidate_frames = list(itemgetter(*segment_idxs)(frames))
        print('CAndidate frames  len({}), indexes={}'.format(len(candidate_frames),segment_idxs))

        dyn_image, _ = getDynamicImage(frames[0:30]) #all frames
        dyn_image = extractor.__format_dynamic_image__(dyn_image)
        cv2.imshow("dyn_image", dyn_image)
        key = cv2.waitKey(0)

        dyn_image_cand, _ = getDynamicImage(candidate_frames) #candidate frames
        dyn_image_cand = extractor.__format_dynamic_image__(dyn_image_cand)
        cv2.imshow("dyn_image_cand", dyn_image_cand)
        key = cv2.waitKey(0)

        
        

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