import numpy as np
import lijnn
from lijnn import cuda, utils
from lijnn.core import Function, Variable, as_variable, as_array


def classification(y, t):
    y, t = as_variable(y), as_variable(t)

    pred = y.data.argmax(axis=1)
    result = (pred == t.data)
    acc = result.mean()
    return Variable(as_array(acc))


def dectection(y, x_bbox, t, t_bbox):
    y, t = as_variable(y), as_variable(t)

    pred = y.data.argmax(axis=1)
    result = (pred == t.data)
    acc = result.mean()
    return Variable(as_array(acc))
