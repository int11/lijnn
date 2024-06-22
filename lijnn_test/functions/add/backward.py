from lijnn import *
from lijnn.core import add
from lijnn_test.functions.main_backward import test_function_backward

def test():
    input_data = (3,3)
    backward = (1,1)
    test_function_backward(input_data, 6, backward, add)



if __name__ == '__main__':
    test()