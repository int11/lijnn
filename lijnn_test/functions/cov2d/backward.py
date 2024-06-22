import lijnn
from lijnn_test.functions.unit_test import f_unit_test, f_unit_test_withTorch
import numpy as np


def test():
    input_data_shape = (1,1,4,4)
    input_data = np.arange(np.prod(input_data_shape)).reshape(input_data_shape).astype(np.float32)
    output = np.array([[[[ 5,  6,  7],
         [ 9, 10, 11],
         [13, 14, 15]]]])
    
    backward = np.array([[[[0, 0, 0, 0],
         [0, 1, 1, 1],
         [0, 1, 1, 1],
         [0, 1, 1, 1]]]])
    
    f_unit_test(input_data, output, backward, lijnn.functions.conv2d, 2)

def test_torch():
    import torch.nn.functional as F

    N, C, H, W = 1, 1, 10, 10
    OC, K = 1, 3
    input_data_shape = (N, C, H, W)
    w_shape = (OC, C, K, K)
    b_shape = (OC, )
    kargs = {"stride":2}
    f_unit_test_withTorch((input_data_shape, w_shape, b_shape), F.conv2d, lijnn.functions.conv2d, **kargs)
    
if __name__ == "__main__":
    test_torch()