import lijnn
import numpy as np
from lijnn.utils import *
import lijnn.functions_conv as F
from lijnn_test.im2col_speed.main import *

class Conv2d_O(lijnn.Function):
    def __init__(self, stride=1, pad=0):
        super().__init__()
        self.stride = pair(stride)
        self.pad = pair(pad)

    def forward(self, x, W, b):
        """simple

        Weight = W
        N, C, H, W = x.shape
        OC, C, KH, KW = Weight.shape
        SH, SW = pair(self.stride)
        PH, PW = pair(self.pad)
        OH = get_conv_outsize(H, KH, SH, PH)
        OW = get_conv_outsize(W, KW, SW, PW)

        col = im2col(x, (KH, KW), self.stride, self.pad, to_matrix=True)
        Weight = Weight.reshape(OC, -1).transpose()
        t = linear(col, Weight, b)
        y = t.reshape(N, OH, OW, OC).transpose(0, 3, 1, 2)
        """

        xp = cuda.get_array_module(x)

        KH, KW = W.shape[2:]
        col = im2col_stride_O(x, (KH, KW), self.stride, self.pad, to_matrix=False)

        y = xp.tensordot(col, W, ((1, 4, 5), (1, 2, 3)))
        if b is not None:
            y += b
        y = xp.rollaxis(y, 3, 1)
        return y



def conv2d_O(x, W, b=None, stride=1, pad=0):
    return Conv2d_O(stride, pad)(x, W, b)


class Conv2d_K(Conv2d_O):
    def forward(self, x, W, b):
        """simple

        Weight = W
        N, C, H, W = x.shape
        OC, C, KH, KW = Weight.shape
        SH, SW = pair(self.stride)
        PH, PW = pair(self.pad)
        OH = get_conv_outsize(H, KH, SH, PH)
        OW = get_conv_outsize(W, KW, SW, PW)

        col = im2col(x, (KH, KW), self.stride, self.pad, to_matrix=True)
        Weight = Weight.reshape(OC, -1).transpose()
        t = linear(col, Weight, b)
        y = t.reshape(N, OH, OW, OC).transpose(0, 3, 1, 2)
        """

        xp = cuda.get_array_module(x)

        KH, KW = W.shape[2:]
        col = im2col_stride_K(x, (KH, KW), self.stride, self.pad, to_matrix=False)


        y = xp.tensordot(col, W, ((1, 2, 3), (1, 2, 3)))
        if b is not None:
            y += b
        y = xp.rollaxis(y, 3, 1)
        return y



def conv2d_K(x, W, b=None, stride=1, pad=0):
    return Conv2d_K(stride, pad)(x, W, b)

n, c, h, w = 50, 100, 100, 100
oc, kernel_size, = 100, 3

a = Variable(np.random.randn(n, c, h, w).astype(np.float32))
b = Variable(np.random.randn(oc, c, kernel_size, kernel_size).astype(np.float32))


function = [conv2d_K, conv2d_O]


timer = Timer()

for f in function:
    time_stack = timer.function_speed_check(10, f, a, b, stride=1, pad=0)
