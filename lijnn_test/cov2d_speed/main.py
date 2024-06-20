import lijnn
import numpy as np
from lijnn.utils import *
import lijnn.functions_conv as F
from test1.im2col_speed.main import im2col_array1

class Conv2d1(lijnn.Function):
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
        col = im2col_array1(x, (KH, KW), self.stride, self.pad, to_matrix=False)

        y = xp.tensordot(col, W, ((1, 4, 5), (1, 2, 3)))
        if b is not None:
            y += b
        y = xp.rollaxis(y, 3, 1)
        return y



def conv2d1(x, W, b=None, stride=1, pad=0):
    return Conv2d1(stride, pad)(x, W, b)


class Conv2d2(lijnn.Function):
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
        col = im2col_array2(x, (KH, KW), self.stride, self.pad, to_matrix=False)


        y = xp.tensordot(col, W, ((1, 2, 3), (1, 2, 3)))
        if b is not None:
            y += b
        y = xp.rollaxis(y, 3, 1)
        return y



def conv2d2(x, W, b=None, stride=1, pad=0):
    return Conv2d2(stride, pad)(x, W, b)

n, c, h, w = 1, 100, 15, 15
oc, kh, kw = 50, 10, 10

a = Variable(np.random.randn(n, c, h, w).astype(np.float32))
b = Variable(np.random.randn(oc, c, kh, kw).astype(np.float32))

with Timer() as t:
  y = conv2d1(a, b)

a = Variable(np.random.randn(n, c, h, w).astype(np.float32))
b = Variable(np.random.randn(oc, c, kh, kw).astype(np.float32))

with Timer() as t:
  y = conv2d2(a, b)