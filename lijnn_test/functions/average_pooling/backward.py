import lijnn
from lijnn_test.functions.main_backward import test_function_backward
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
    
    test_function_backward(input_data, output, backward, lijnn.functions.average_pooling, 2)

def test_torch():
    import torch
    import torch.nn.functional as F

    input_data_shape = (3,2,7,7)
    kargs = {"kernel_size":3, "stride":2}
    input_data = np.arange(np.prod(input_data_shape)).reshape(input_data_shape).astype(np.float32)
    
    input_tensor = torch.tensor(input_data, requires_grad=True)
    
    output = F.avg_pool2d(input_tensor, **kargs)

    output.backward(torch.ones_like(output))
    output = output.detach().numpy()
    backward = input_tensor.grad.numpy()
    
    test_function_backward(input_data, output, backward, lijnn.functions.average_pooling, **kargs)
    
if __name__ == "__main__":
    test_torch()