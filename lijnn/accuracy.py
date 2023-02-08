from lijnn import utils
from lijnn.core import Variable
from lijnn.cuda import as_numpy

# accuracy function must return dictionary
def classification(y, t):
    y, t = as_numpy(y), as_numpy(t)
    acc = (y.argmax(axis=1) == t).mean()
    return {'acc': Variable(as_numpy(acc))}


def dectection(y, x_bbox, t, t_bbox):
    y, x_bbox, t, t_bbox = as_numpy(y), as_numpy(x_bbox), as_numpy(t), as_numpy(t_bbox)

    acc = (y.argmax(axis=1) == t).mean()
    iou = sum([utils.IOU(a, b) for a, b in zip(x_bbox, t_bbox)])
    return {'acc': Variable(as_numpy(acc)), 'iou': Variable(as_numpy(iou))}
