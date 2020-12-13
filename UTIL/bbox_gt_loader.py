import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import re
import constants
import json
from PREPROCESING.segmentation import bbox_from_di

def load_bbox_gt(video_pth, label, paths):
    _, video_name = os.path.split(video_pth)
    # if label==1:
    video_name = video_name[:-8]
    bdx_file_path = os.path.join(constants.PATH_UCFCRIME2LOCAL_Txt_ANNOTATIONS, video_name+'.txt')
    # print('bdx_file_path=', label, bdx_file_path)
    data = []
    with open(bdx_file_path, 'r') as file:
        for row in file:
            data.append(row.split())
    # data = np.array(data)
    gt_bboxes = []
    # print('len(paths[0][0])=',len(paths[0][0]))
    for i,frame_path in enumerate(paths):
        # print('------frame_path=',frame_path)
        pth, frame_name = os.path.split(frame_path)
        splits = re.split('(\d+)', frame_name)
        frame_number = int(splits[1])
        frame_data = data[frame_number]
        # print('video={}, frame={}, frame_number={}, gt={}'.format(video_name, frame_name, frame_number, frame_data))
        if frame_number != int(frame_data[5]):
            print('=========*********** Error en Ground Truth!!!!!!!!!')
            break
        x0, y0, w, h = int(frame_data[1]), int(frame_data[2]), int(frame_data[3])-int(frame_data[1]), int(frame_data[4])-int(frame_data[2])
        gt_bboxes.append([x0, y0, w, h])

    one_box = None
    for gtb in gt_bboxes:
        if one_box is None:
            one_box = gtb
            # xmin, ymin, w, h = one_box
        else:
            xmin = min(gtb[0], one_box[0])
            ymin = min(gtb[1], one_box[1])
            xmax = max(gtb[0]+gtb[2], one_box[0]+one_box[2])
            ymax = max(gtb[1]+gtb[3], one_box[1]+one_box[3])
            one_box = [xmin, ymin, xmax-xmin, ymax-ymin]
    return gt_bboxes, one_box

def pseudo_p_bbox(video_pth, segment, label):
    pth, video_name = os.path.split(video_pth)
    pth, set_pth = os.path.split(pth)
    _, label_pth = os.path.split(pth)
    # video_name = video_name[:-8]
    bdx_file_path = os.path.join(constants.PATH_RWF_2000_ROIS, label_pth, set_pth, video_name)
    # print(bdx_file_path)
    dict_list = json.load(open(bdx_file_path, "r"))
    segment_bboxes = []
    for i,d in enumerate(dict_list):
        for f in segment:
            if d['file'] == f[:-4]:
                segment_bboxes.append(d['file'])
    # print(dict_list[0])
    return segment_bboxes

def intersection(m_box, p_box):
    x0, y0, w0, h0 = m_box
    x1, y1, w1, h1 = p_box
    dx = int(min(x0+w0, x0+w1) - max(x0, x1))
    dy = int(min(y0+h0, y1+h1) - max(y0, y1))
    if (dx>=0) and (dy>=0):
        return dx * dy
    else:
        return 0
def union(m_box, p_box):
    x0 = min(m_box[0], p_box[0])
    y0 = min(m_box[1], p_box[1])
    x1 = max(m_box[0]+m_box[2], p_box[0]+p_box[2])
    y1 = max(m_box[1]+m_box[3], p_box[1]+p_box[3])
    return x0,y0,x1,y1

def m_x_p_pseudo_bbox(m_bboxes, p_bboxes):
    m_x_box = None
    for m_box in m_bboxes:
        m_area = m_box[2]*m_box[3]
        motion_persons = []
        for p_box in p_bboxes:
            p_area = p_box[2]*p_box[3]
            it = intersection(m_box, p_box)
            if it >= p_area/2 or it >= m_area/2:
                motion_persons.append(p_box)
    
if __name__ == "__main__":
    video = '/Users/davidchoqueluqueroman/Desktop/PROJECTS-SOURCE-CODES/violencedetection2/DATASETS/RWF-2000/frames/train/Fight/_2RYnSFPD_U_0'
    frames = os.listdir(video)
    frames.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    frames = frames[0:10]
    segment_bboxes = pseudo_bbox(video,frames,1)
    print(segment_bboxes)
