import os
import re
import constants

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
