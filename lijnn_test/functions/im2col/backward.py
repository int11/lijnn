from lijnn import *
import lijnn
from lijnn_test.functions.unit_test import f_unit_test
import numpy as np

def test():
    input_data_shape = (1,1,4,4)
    input_data = np.arange(np.prod(input_data_shape)).reshape(input_data_shape)
    output = np.array([[[[[[ 0,  1],
              [ 4,  5]],
         
             [[ 1,  2],
              [ 5,  6]],
         
             [[ 2,  3],
              [ 6,  7]]],
         
         
            [[[ 4,  5],
              [ 8,  9]],
         
             [[ 5,  6],
              [ 9, 10]],
         
             [[ 6,  7],
              [10, 11]]],
         
         
            [[[ 8,  9],
              [12, 13]],
         
             [[ 9, 10],
              [13, 14]],
         
             [[10, 11],
              [14, 15]]]]]])
    
    backward = np.array([[[[1, 2, 2, 1],
                [2, 4, 4, 2],
                [2, 4, 4, 2],
                [1, 2, 2, 1]]]])
    f_unit_test(input_data, output, backward, lijnn.functions.im2col, 2, to_matrix=False)


def test_torch():
    f_unit_test_withTorch((input_data_shape, w_shape, b_shape), F.conv2d, lijnn.functions.im2col, **kargs)
if __name__ == '__main__':
    test()