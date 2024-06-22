import lijnn
from lijnn_test.functions.unit_test import f_unit_test
import numpy as np

def test():
    input_data_shape = (1,1,4,4)
    input_data = np.arange(np.prod(input_data_shape)).reshape(input_data_shape)
    output = np.array([[[[ 5,  6,  7],
         [ 9, 10, 11],
         [13, 14, 15]]]])
    
    backward = np.array([[[[0, 0, 0, 0],
         [0, 1, 1, 1],
         [0, 1, 1, 1],
         [0, 1, 1, 1]]]])
    
    f_unit_test(input_data, output, backward, lijnn.functions.max_pooling, 2)


if __name__ == "__main__":
    test()