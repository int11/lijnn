from lijnn import *
import numpy as np
import cupy as cp
from lijnn.functions import *

def test(gpu):
    x = Variable(np.ones([1,1,4,4]))
    if gpu:
        x.to_gpu()
    y = im2col(x, 2)

    y.backward(retain_grad=True)
    x.grad.to_cpu()
    return np.array_equal(x.grad.data, 
            np.array([[[[1, 2, 2, 1],
                        [2, 4, 4, 2],
                        [2, 4, 4, 2],
                        [1, 2, 2, 1]]]]))

if __name__ == '__main__':
    print(test(False))
    print(test(True))