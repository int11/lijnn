import numpy as np
import lijnn
from lijnn import cuda, utils
from lijnn.core import Function, Variable, as_variable, as_array


def classification(y, t):
    y, t = as_array(y), as_array(t)
    acc = (y.argmax(axis=1) == t).mean()
    return {'acc': Variable(as_array(acc))}


def dectection(y, x_bbox, t, t_bbox):
    y, x_bbox, t, t_bbox = as_array(y), as_array(x_bbox), as_array(t), as_array(t_bbox)

    acc = (y.argmax(axis=1) == t).mean()
    iou = sum([utils.IOU(a, b) for a, b in zip(x_bbox, t_bbox)])
    return {'acc': Variable(as_array(acc)), 'iou': Variable(as_array(iou))}
