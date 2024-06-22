from lijnn import *
from lijnn.core import add
from lijnn_test.functions.unit_test import f_unit_test

def test():
    input_data = (3,3)
    backward = (1,1)
    f_unit_test(input_data, 6, backward, add)



if __name__ == '__main__':
    test()