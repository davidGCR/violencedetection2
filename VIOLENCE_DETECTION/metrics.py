


def intersection(x0, y0, w0, h0, x1, y1, w1, h1): 
    dx = int(min(x0+w0, x0+w1) - max(x0, x1))
    dy = int(min(y0+h0, y1+h1) - max(y0, y1))
    if (dx>=0) and (dy>=0):
        return dx * dy
    else: return 0

def iou(x0, y0, w0, h0, x1, y1, w1, h1):
    # calculate area of intersection rectangle
    inter_area = intersection(x0, y0, w0, h0, x1, y1, w1, h1)
    # calculate area of actual and predicted boxes
    gt_area = w0*h0
    pred_area = w1*h1
 
    # computing intersection over union
    m = inter_area / float(gt_area + pred_area - inter_area)
 
    # return the intersection over union value
    return m


def loc_error(y, y_pred, iou_thresh=0.5):
    counter = 0
    for i, gtb in enumerate(y):
        # print(y[i][0],y_pred[i][0])
        yy = y[i][0]
        yy_pred = y_pred[i][0]
        if yy != yy_pred:
            counter +=1
        elif yy==1:
            x0, y0, w0, h0 = y[i][1:]
            x1, y1, w1, h1 = y_pred[i][1:]
            m = iou(x0, y0, w0, h0, x1, y1, w1, h1)
            if m < iou_thresh:
                counter += 1
    el = counter / len(y)
    return el

