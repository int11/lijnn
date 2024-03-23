import numpy as np
from lijnn import cuda
from lijnn.core import Function
from lijnn.utils import pair, get_conv_outsize

class FinePooling(Function):
    def __init__(self, kernel_size, to_batch):
        self.kernel_size = kernel_size
        self.to_batch = to_batch

    def forward(self, x):
        N, C, H, W = x.shape
        KH, KW = pair(self.kernel_size)
        SH, SW = pair(self.kernel_size)
        OH = get_conv_outsize(H, KH, SH, 0)
        OW = get_conv_outsize(W, KW, SW, 0)

        xp = cuda.get_array_module(x)

        strides = x.strides
        col = xp.lib.stride_tricks.as_strided(x, (3, 3, N, C, KH, KW, OH, OW),
                                              (strides[2], strides[3], strides[0], strides[1], strides[2], strides[3],
                                               strides[2] * SH, strides[3] * SW))
        col = col.reshape(3, 3, N, C, KH * KW, OH, OW)
        self.indexes = col.argmax(axis=4)
        y = col.max(axis=4)
        return y.reshape(3 * 3 * N, C, OH, OW) if self.to_batch else y

    def backward(self, gy):
        return FinePoolingGrad(self, self.to_batch)(gy)


class FinePoolingGrad(Function):
    def __init__(self, fpool2d, to_batch):
        self.mpool2d = fpool2d
        self.kernel_size = fpool2d.kernel_size
        self.input_shape = fpool2d.inputs[0].shape
        self.dtype = fpool2d.inputs[0].dtype
        self.indexes = fpool2d.indexes
        self.to_batch = to_batch

    def forward(self, gy):
        xp = cuda.get_array_module(gy)

        N, C, H, W = self.input_shape
        KH, KW = pair(self.kernel_size)
        SH, SW = pair(self.kernel_size)
        PH, PW = pair(0)
        OH = get_conv_outsize(H, KH, SH, PH)
        OW = get_conv_outsize(W, KW, SW, PW)
        gy = gy.reshape(3, 3, N, C, OH, OW) if self.to_batch else gy
        _, _, N, C, OH, OW = gy.shape

        gcol = xp.zeros((3 * 3 * N * C * OH * OW * KH * KW), dtype=self.dtype)

        indexes = (self.indexes.ravel()
                   + xp.arange(0, self.indexes.size * KH * KW, KH * KW))

        gcol[indexes] = gy.ravel()
        gcol = gcol.reshape(3, 3, N, C, OH, OW, KH, KW)
        gcol = xp.swapaxes(gcol, 4, 6)
        gcol = xp.swapaxes(gcol, 5, 7)

        img = np.zeros((N, C, H, W), dtype=gcol.dtype)

        for FH in range(3):
            for FW in range(3):
                for j in range(KH):
                    j_lim = j + SH * OH
                    for i in range(KW):
                        i_lim = i + SW * OW
                        img[:, :, j + FH:j_lim:SH, i + FW:i_lim:SW] += gcol[FH, FW, :, :, j, i, :, :]
        return img[:, :, PH:H + PH, PW:W + PW]


def find_pooling(x, kernel_size, to_batch=True):
    return FinePooling(kernel_size, to_batch)(x)