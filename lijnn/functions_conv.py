import numpy as np
from lijnn import cuda
from lijnn.core import Function, as_variable
from lijnn.utils import pair, get_conv_outsize, get_deconv_outsize
from lijnn.functions import linear, broadcast_to


# =============================================================================
#  conv2d / deconv2d
# =============================================================================
class Conv2d(Function):
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
        col = im2col_array(x, (KH, KW), self.stride, self.pad, to_matrix=False)

        y = xp.tensordot(col, W, ((1, 2, 3), (1, 2, 3)))
        if b is not None:
            y += b
        y = xp.rollaxis(y, 3, 1)
        return y

    def backward(self, gy):
        x, W, b = self.inputs
        # ==== gx ====
        gx = deconv2d(gy, W, b=None, stride=self.stride, pad=self.pad,
                      outsize=(x.shape[2], x.shape[3]))
        # ==== gW ====
        gW = Conv2DGradW(self)(x, gy)
        # ==== gb ====
        gb = None
        if b.data is not None:
            gb = gy.sum(axis=(0, 2, 3))
        return gx, gW, gb


def conv2d(x, W, b=None, stride=1, pad=0):
    return Conv2d(stride, pad)(x, W, b)


class Deconv2d(Function):
    def __init__(self, stride=1, pad=0, outsize=None):
        super().__init__()
        self.stride = pair(stride)
        self.pad = pair(pad)
        self.outsize = outsize

    def forward(self, gy, W, b):
        xp = cuda.get_array_module(gy)

        Weight = W
        SH, SW = self.stride
        PH, PW = self.pad
        C, OC, KH, KW = Weight.shape
        N, C, H, W = gy.shape
        if self.outsize is None:
            out_h = get_deconv_outsize(H, KH, SH, PH)
            out_w = get_deconv_outsize(W, KW, SW, PW)
        else:
            out_h, out_w = pair(self.outsize)
        img_shape = (N, OC, out_h, out_w)

        gcol = xp.tensordot(Weight, gy, (0, 1))
        gcol = xp.rollaxis(gcol, 3)
        gx = col2im_array(gcol, img_shape, (KH, KW), self.stride, self.pad,
                          to_matrix=False)
        # b, k, h, w
        if b is not None:
            self.no_bias = True
            gx += b.reshape((1, b.size, 1, 1))
        return gx

    def backward(self, gy):
        x, W, b = self.inputs

        # ==== gx ====
        gx = conv2d(gy, W, b=None, stride=self.stride, pad=self.pad)
        # ==== gW ====
        f = Conv2DGradW(self)
        gW = f(gy, x)
        # ==== gb ====
        gb = None
        if b.data is not None:
            gb = gy.sum(axis=(0, 2, 3))
        return gx, gW, gb


def deconv2d(x, W, b=None, stride=1, pad=0, outsize=None):
    return Deconv2d(stride, pad, outsize)(x, W, b)


class Conv2DGradW(Function):
    def __init__(self, conv2d):
        W = conv2d.inputs[1]
        kh, kw = W.shape[2:]
        self.kernel_size = (kh, kw)
        self.stride = conv2d.stride
        self.pad = conv2d.pad

    def forward(self, x, gy):
        xp = cuda.get_array_module(x)

        col = im2col_array(x, self.kernel_size, self.stride, self.pad,
                           to_matrix=False)
        gW = xp.tensordot(gy, col, ((0, 2, 3), (0, 4, 5)))
        return gW

    def backward(self, gys):
        x, gy = self.inputs
        gW, = self.outputs

        xh, xw = x.shape[2:]
        gx = deconv2d(gy, gW, stride=self.stride, pad=self.pad,
                      outsize=(xh, xw))
        ggy = conv2d(x, gW, stride=self.stride, pad=self.pad)
        return gx, ggy


class MaxPooling(Function):
    def __init__(self, kernel_size, stride=1, pad=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        """simple

        N, C, H, W = x.shape
        KH, KW = pair(self.kernel_size)
        PH, PW = pair(self.pad)
        SH, SW = pair(self.stride)
        OH = get_conv_outsize(H, KH, SH, PH)
        OW = get_conv_outsize(W, KW, SW, PW)

        col = im2col(x, self.kernel_size, self.stride, self.pad, to_matrix=True)
        col = col.reshape(-1, KH * KW)
        y = col.max(axis=1)
        y = y.reshape(N, OH, OW, C).transpose(0, 3, 1, 2)
        """

        col = im2col_array(x, self.kernel_size, self.stride, self.pad,
                           to_matrix=False)

        N, C, KH, KW, OH, OW = col.shape
        col = col.reshape(N, C, KH * KW, OH, OW)
        self.indexes = col.argmax(axis=2)
        y = col.max(axis=2)
        return y

    def backward(self, gy):
        return MaxPoolingGrad(self)(gy)


class MaxPoolingGrad(Function):
    def __init__(self, mpool2d):
        self.mpool2d = mpool2d
        self.kernel_size = mpool2d.kernel_size
        self.stride = mpool2d.stride
        self.pad = mpool2d.pad
        self.input_shape = mpool2d.inputs[0].shape
        self.dtype = mpool2d.inputs[0].dtype
        self.indexes = mpool2d.indexes

    def forward(self, gy):
        xp = cuda.get_array_module(gy)

        N, C, OH, OW = gy.shape
        N, C, H, W = self.input_shape
        KH, KW = pair(self.kernel_size)

        gcol = xp.zeros((N * C * OH * OW * KH * KW), dtype=self.dtype)

        indexes = (self.indexes.ravel()
                   + xp.arange(0, self.indexes.size * KH * KW, KH * KW))

        gcol[indexes] = gy.ravel()
        gcol = gcol.reshape(N, C, OH, OW, KH, KW)
        gcol = xp.swapaxes(gcol, 2, 4)
        gcol = xp.swapaxes(gcol, 3, 5)

        gx = col2im_array(gcol, (N, C, H, W), self.kernel_size, self.stride,
                          self.pad, to_matrix=False)
        return gx

    def backward(self, ggx):
        f = MaxPoolingWithIndexes(self.mpool2d)
        return f(ggx)


class MaxPoolingWithIndexes(Function):
    def __init__(self, mpool2d):
        self.kernel_size = mpool2d.kernel_size
        self.stride = mpool2d.stride
        self.pad = mpool2d.pad
        self.input_shpae = mpool2d.inputs[0].shape
        self.dtype = mpool2d.inputs[0].dtype
        self.indexes = mpool2d.indexes

    def forward(self, x):
        col = im2col_array(x, self.kernel_size, self.stride, self.pad,
                           to_matrix=False)
        N, C, KH, KW, OH, OW = col.shape
        col = col.reshape(N, C, KH * KW, OH, OW)
        col = col.transpose(0, 1, 3, 4, 2).reshape(-1, KH * KW)
        indexes = self.indexes.ravel()
        col = col[np.arange(len(indexes)), indexes]
        return col.reshape(N, C, OH, OW)


def max_pooling(x, kernel_size, stride=1, pad=0):
    return MaxPooling(kernel_size, stride, pad)(x)


class AveragePooling(Function):
    def __init__(self, kernel_size, stride=1, pad=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.input_shape = None

    def forward(self, x):
        self.input_shape = x.shape
        col = im2col_array(x, self.kernel_size, self.stride, self.pad,
                           to_matrix=False)
        y = col.mean(axis=(2, 3))
        return y

    def backward(self, gy):
        # TODO(Koki): This is simple implementation
        N, C, OH, OW = gy.shape
        KW, KH = pair(self.kernel_size)
        gy /= (KW * KH)
        gcol = broadcast_to(gy.reshape(-1), (KH, KW, N * C * OH * OW))
        gcol = gcol.reshape(KH, KW, N, C, OH, OW).transpose(2, 3, 0, 1, 4, 5)
        gx = col2im(gcol, self.input_shape, self.kernel_size, self.stride,
                    self.pad, to_matrix=False)
        return gx


def average_pooling(x, kernel_size, stride=1, pad=0):
    return AveragePooling(kernel_size, stride, pad)(x)


class FinePooling(Function):
    def __init__(self, kernel_size, to_batch):
        super().__init__()
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
        col = xp.lib.stride_tricks.as_strided(x, (3, 3, N, C,  KH, KW, OH, OW),
                                              (strides[2], strides[3], strides[0], strides[1], strides[2], strides[3],
                                               strides[2] * SH, strides[3] * SW))
        col = col.reshape(3, 3, N, C,  KH * KW, OH, OW)
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


# =============================================================================
#  im2col / col2im
# =============================================================================
class Im2col(Function):
    def __init__(self, kernel_size, stride, pad, to_matrix):
        super().__init__()
        self.input_shape = None
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.to_matrix = to_matrix

    def forward(self, x):
        self.input_shape = x.shape
        y = im2col_array(x, self.kernel_size, self.stride, self.pad,
                         self.to_matrix)
        return y

    def backward(self, gy):
        gx = col2im(gy, self.input_shape, self.kernel_size, self.stride,
                    self.pad, self.to_matrix)
        return gx


def im2col(x, kernel_size, stride=1, pad=0, to_matrix=True):
    """Extract patches from an image based on the filter.

    Args:
        x (`dezero.Variable` or `ndarray`): Input variable of shape
            `(N, C, H, W)`
        kernel_size (int or (int, int)): Size of kernel.
        stride (int or (int, int)): Stride of kernel.
        pad (int or (int, int)): Spatial padding width for input arrays.
        to_matrix (bool): If True the `col` will be reshaped to 2d array whose
            shape is `(N*OH*OW, C*KH*KW)`

    Returns:
        `dezero.Variable`: Output variable. If the `to_matrix` is False, the
            output shape is `(N, C, KH, KW, OH, OW)`, otherwise
            `(N*OH*OW, C*KH*KW)`.

    Notation:
    - `N` is the batch size.
    - `C` is the number of the input channels.
    - `H` and `W` are the height and width of the input image, respectively.
    - `KH` and `KW` are the height and width of the filters, respectively.
    - `SH` and `SW` are the strides of the filter.
    - `PH` and `PW` are the spatial padding sizes.
    - `OH` and `OW` are the the height and width of the output, respectively.
    """
    y = Im2col(kernel_size, stride, pad, to_matrix)(x)
    return y


class Col2im(Function):
    def __init__(self, input_shape, kernel_size, stride, pad, to_matrix):
        super().__init__()
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.to_matrix = to_matrix

    def forward(self, x):
        y = col2im_array(x, self.input_shape, self.kernel_size, self.stride,
                         self.pad, self.to_matrix)
        return y

    def backward(self, gy):
        gx = im2col(gy, self.kernel_size, self.stride, self.pad,
                    self.to_matrix)
        return gx


def col2im(x, input_shape, kernel_size, stride=1, pad=0, to_matrix=True):
    return Col2im(input_shape, kernel_size, stride, pad, to_matrix)(x)


# =============================================================================
#  numpy im2col
# =============================================================================

def im2col_array(img, kernel_size, stride, pad, to_matrix=True):
    N, C, H, W = img.shape
    KH, KW = pair(kernel_size)
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    xp = cuda.get_array_module(img)
    if xp != np:
        col = _im2col_gpu(img, kernel_size, stride, pad)
    else:

        img = np.pad(img,
                     ((0, 0), (0, 0), (PH, PH + SH - 1), (PW, PW + SW - 1)),
                     mode='constant', constant_values=(0,))
        # col = np.ndarray((N, C, KH, KW, OH, OW), dtype=img.dtype)
        # for j in range(KH):
        #     j_lim = j + SH * OH
        #     for i in range(KW):
        #         i_lim = i + SW * OW
        #         col[:, :, j, i, :, :] = img[:, :, j:j_lim:SH, i:i_lim:SW]

        strides = img.strides
        col = xp.lib.stride_tricks.as_strided(img, (N, C, KH, KW, OH, OW), (
            strides[0], strides[1], strides[2], strides[3], strides[2] * SH, strides[3] * SW))

    if to_matrix:
        col = col.transpose((0, 4, 5, 1, 2, 3)).reshape((N * OH * OW, -1))

    return col


def col2im_array(col, img_shape, kernel_size, stride, pad, to_matrix=True):
    N, C, H, W = img_shape
    KH, KW = pair(kernel_size)
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    if to_matrix:
        col = col.reshape(N, OH, OW, C, KH, KW).transpose(0, 3, 4, 5, 1, 2)

    xp = cuda.get_array_module(col)
    if xp != np:
        img = _col2im_gpu(col, SH, SW, PH, PW, H, W)
        return img
    else:
        img = np.zeros((N, C, H + 2 * PH + SH - 1, W + 2 * PW + SW - 1),
                       dtype=col.dtype)
        for j in range(KH):
            j_lim = j + SH * OH
            for i in range(KW):
                i_lim = i + SW * OW
                img[:, :, j:j_lim:SH, i:i_lim:SW] += col[:, :, j, i, :, :]
        return img[:, :, PH:H + PH, PW:W + PW]


def _im2col_gpu(img, kernel_size, stride, pad):
    """im2col function for GPU.
    This code is ported from Chainer:
    https://github.com/chainer/chainer/blob/v6.4.0/chainer/utils/conv.py
    """
    n, c, h, w = img.shape
    kh, kw = pair(kernel_size)
    sy, sx = pair(stride)
    ph, pw = pair(pad)
    out_h = get_conv_outsize(h, kh, sy, ph)
    out_w = get_conv_outsize(w, kw, sx, pw)
    dy, dx = 1, 1
    col = cuda.cupy.empty((n, c, kh, kw, out_h, out_w), dtype=img.dtype)

    cuda.cupy.ElementwiseKernel(
        'raw T img, int32 h, int32 w, int32 out_h, int32 out_w,'
        'int32 kh, int32 kw, int32 sy, int32 sx, int32 ph, int32 pw,'
        'int32 dy, int32 dx',
        'T col',
        '''
           int c0 = i / (kh * kw * out_h * out_w);
           int ky = i / (kw * out_h * out_w) % kh;
           int kx = i / (out_h * out_w) % kw;
           int out_y = i / out_w % out_h;
           int out_x = i % out_w;
           int in_y = ky * dy + out_y * sy - ph;
           int in_x = kx * dx + out_x * sx - pw;
           if (in_y >= 0 && in_y < h && in_x >= 0 && in_x < w) {
             col = img[in_x + w * (in_y + h * c0)];
           } else {
             col = 0;
           }
        ''',
        'im2col')(img.reduced_view(), h, w, out_h, out_w,
                  kh, kw, sy, sx, ph, pw,
                  dy, dx, col)

    return col


def _col2im_gpu(col, sy, sx, ph, pw, h, w):
    """col2im function for GPU.
    This code is ported from Chainer:
    https://github.com/chainer/chainer/blob/v6.4.0/chainer/utils/conv.py
    """
    n, c, kh, kw, out_h, out_w = col.shape
    dx, dy = 1, 1
    img = cuda.cupy.empty((n, c, h, w), dtype=col.dtype)

    cuda.cupy.ElementwiseKernel(
        'raw T col, int32 h, int32 w, int32 out_h, int32 out_w,'
        'int32 kh, int32 kw, int32 sy, int32 sx, int32 ph, int32 pw,'
        'int32 dx, int32 dy',
        'T img',
        '''
           int c0 = i / (h * w);
           int y  = i / w % h;
           int x  = i % w;
           T val = 0;
           for (int ky = 0; ky < kh; ++ky) {
             int out_y = (y + ph - ky * dy);
             if (0 > out_y || out_y >= out_h * sy) continue;
             if (out_y % sy != 0) continue;
             out_y /= sy;
             for (int kx = 0; kx < kw; ++kx) {
               int out_x = (x + pw - kx * dx);
               if (0 > out_x || out_x >= out_w * sx) continue;
               if (out_x % sx != 0) continue;
               out_x /= sx;
               int k = out_y + out_h * (kx + kw * (ky + kh * c0));
               val = val + col[out_x + out_w * k];
             }
           }
           img = val;
        ''',
        'col2im')(col.reduced_view(),
                  h, w, out_h, out_w, kh, kw, sy, sx, ph, pw, dx, dy, img)
    return img


def _cu_conv_sum(y, x, n):
    # Convolutional sum
    # TODO(beam2d): Use scan computation
    rdim = x.size // (x.shape[0] * x.shape[1])
    cuda.cupy.ElementwiseKernel(
        'raw T x, int32 rdim, int32 N, int32 n_', 'raw T y',
        '''
          int half_n = n_ / 2;
          int offset = i / rdim * N * rdim + i % rdim;

          float sum_part = 0;
          for (int j = 0; j < N + half_n; ++j) {
            if (j < N) {
              sum_part += x[offset + j * rdim];
            }
            if (j >= n_) {
              sum_part -= x[offset + (j - n_) * rdim];
            }
            if (j >= half_n) {
              y[offset + (j - half_n) * rdim] = sum_part;
            }
          }
        ''', 'lrn_conv_sum')(x, rdim, x.shape[1], n, y,
                             size=x.shape[0] * rdim)


class LocalResponseNormalization(Function):
    """Cross-channel normalization function used in AlexNet."""

    def __init__(self, n=5, k=2, alpha=1e-4, beta=.75):
        self.n = n
        self.k = k
        self.alpha = alpha
        self.beta = beta

        self.scale = None
        self.indexes = None
        self.unit_scale = None

    def forward(self, x):
        xp = cuda.get_array_module(x)
        if xp != np:
            y = self.forward_gpu(x)
        else:
            half_n = self.n // 2
            x2 = np.square(x)
            sum_part = x2.copy()
            for i in range(1, half_n + 1):
                sum_part[:, i:] += x2[:, :-i]
                sum_part[:, :-i] += x2[:, i:]
            self.unit_scale = self.k + self.alpha * sum_part
            self.scale = self.unit_scale ** -self.beta
            y = x * self.scale
        return y

    def forward_gpu(self, x):
        y = cuda.cupy.square(x)  # temporary
        self.scale = cuda.cupy.empty_like(y)
        _cu_conv_sum(self.scale, y, self.n)
        cuda.cupy.ElementwiseKernel(
            'T x, T k, T alpha, T beta',
            'T y, T scale',
            '''scale = k + alpha * scale;
               y = x * pow(scale, -beta);''',
            'lrn_fwd')(x, self.k, self.alpha, self.beta,
                       y, self.scale)
        return y

    def backward(self, gy):
        x = self.inputs[0]
        y = self.outputs[0]()
        f = LocalResponseNormalizationGrad(self.n, self.k, self.alpha, self.beta, self.scale, self.indexes,
                                           self.unit_scale)
        return f(x, y, gy)


class LocalResponseNormalizationGrad(Function):

    def __init__(self, n, k, alpha, beta, scale=None, indexes=None, unit_scale=None):
        self.n = n
        self.k = k
        self.alpha = alpha
        self.beta = beta

        self.scale = scale
        self.indexes = indexes
        self.unit_scale = unit_scale

    def forward(self, x, y, gy):
        xp = cuda.get_array_module(x)
        if xp != np:
            gx = self.forward_gpu(x, y, gy)
        else:

            half_n = self.n // 2
            summand = y * gy / self.unit_scale
            sum_part = summand.copy()
            for i in range(1, half_n + 1):
                sum_part[:, i:] += summand[:, :-i]
                sum_part[:, :-i] += summand[:, i:]

            gx = gy * self.scale - 2 * self.alpha * self.beta * x * sum_part
        return gx

    def forward_gpu(self, x, y, gy):
        summand = cuda.cupy.ElementwiseKernel(
            'T scale, T y, T gy', 'T summand',
            'summand = y * gy / scale',
            'lrn_bwd_summand')(self.scale, y, gy)
        gx = cuda.cupy.empty_like(x)
        _cu_conv_sum(gx, summand, self.n)
        cuda.cupy.ElementwiseKernel(
            ' T x, T gy, T scale, T beta, T coeff', 'T gx',
            'gx = pow(scale, -beta) * gy - coeff * x * gx',
            'lrn_bwd')(x, gy, self.scale,
                       self.beta, 2 * self.alpha * self.beta, gx)
        return gx

    def backward(self, gy):
        # No trivial way to implement double-backward for this function.
        raise NotImplementedError


def local_response_normalization(x, n=5, k=2, alpha=1e-4, beta=.75):
    """Local response normalization across neighboring channels.

    This function implements normalization across channels. Let :math:`x` an
    input image with :math:`N` channels. Then, this function computes an output
    image :math:`y` by following formula:

    .. math::
       y_i = {x_i \\over \\left( k + \\
              \\alpha \\sum_{j=\\max{1, i - n/2}}^{\\min{N, i + n/2}} \\
              x_j^2 \\right)^\\beta}.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.
        n (int): Normalization window width.
        k (float): Smoothing parameter.
        alpha (float): Normalizer scaling parameter.
        beta (float): Normalizer power parameter.

    Returns:
        ~chainer.Variable: Output variable.

    See: Section 3.3 of `ImageNet Classification with Deep Convolutional
    Neural Networks <https://www.cs.toronto.edu/~fritz/absps/imagenet.pdf>`_

    """
    return LocalResponseNormalization(n, k, alpha, beta)(x)
