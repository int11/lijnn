from lijnn import *
import numpy as np
import cupy as cp
from lijnn.functions import *

def test(gpu):
    x = Variable(np.array([3]))
    if gpu:
        x.to_gpu()
    y = add(x, x)

    y.backward(retain_grad=True)

    return (y.grad.data == 1 and x.grad.data == 2)
