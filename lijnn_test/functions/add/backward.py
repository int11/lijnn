from lijnn import *
import numpy as np
import cupy as cp
from lijnn.functions import *
from lijnn_test.functions.main_backward import test_function_backward

def test():
    input_data = (np.array([3]), np.array([3]))
    return test_function_backward(input_data, 1, 2, add, 3, 3)


if __name__ == '__main__':
    print(test())
