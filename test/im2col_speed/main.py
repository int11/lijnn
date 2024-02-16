import numpy as np
import time
from lijnn.utils import pair, get_conv_outsize


def a(img, kernel_size, stride, pad):
    N, C, H, W = img.shape
    KH, KW = pair(kernel_size)
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)
    return N, C, H, W, KH, KW, SH, SW, PH, PW, OH, OW


def im2col_for_O(img, kernel_size, stride, pad):
    N, C, H, W, KH, KW, SH, SW, PH, PW, OH, OW = a(img, kernel_size, stride, pad)

    img = np.pad(img, ((0, 0), (0, 0), (PH, PH + SH - 1), (PW, PW + SW - 1)), mode='constant', constant_values=(0,))
    col = np.ndarray((N, C, OH, OW, KH, KW), dtype=img.dtype)
    for j in range(OH):
        h_start = j * SH
        for i in range(OW):
            w_start = i * SW
            col[:, :, j, i, :, :] = img[:, :, h_start:h_start + KH, w_start:w_start + KW]

    col = col.transpose((0, 2, 3, 1, 4, 5)).reshape((N * OH * OW, -1))

    return col


def im2col_for_K(img, kernel_size, stride, pad):
    N, C, H, W, KH, KW, SH, SW, PH, PW, OH, OW = a(img, kernel_size, stride, pad)

    img = np.pad(img, ((0, 0), (0, 0), (PH, PH + SH - 1), (PW, PW + SW - 1)), mode='constant', constant_values=(0,))
    col = np.ndarray((N, C, KH, KW, OH, OW), dtype=img.dtype)
    for j in range(KH):
        j_lim = j + SH * OH
        for i in range(KW):
            i_lim = i + SW * OW
            col[:, :, j, i, :, :] = img[:, :, j:j_lim:SH, i:i_lim:SW]

    col = col.transpose((0, 4, 5, 1, 2, 3)).reshape((N * OH * OW, -1))

    return col


def im2col_stride_O(img, kernel_size, stride, pad):
    N, C, H, W, KH, KW, SH, SW, PH, PW, OH, OW = a(img, kernel_size, stride, pad)

    img = np.pad(img, ((0, 0), (0, 0), (PH, PH + SH - 1), (PW, PW + SW - 1)), mode='constant', constant_values=(0,))

    strides = img.strides
    col = np.lib.stride_tricks.as_strided(img, (N, C, OH, OW, KH, KW), (
        strides[0], strides[1], strides[2] * SH, strides[3] * SW, strides[2], strides[3]))

    col = col.transpose((0, 2, 3, 1, 4, 5)).reshape((N * OH * OW, -1))

    return col


def im2col_stride_K(img, kernel_size, stride, pad):
    N, C, H, W, KH, KW, SH, SW, PH, PW, OH, OW = a(img, kernel_size, stride, pad)

    img = np.pad(img, ((0, 0), (0, 0), (PH, PH + SH - 1), (PW, PW + SW - 1)), mode='constant', constant_values=(0,))

    strides = img.strides
    col = np.lib.stride_tricks.as_strided(img, (N, C, KH, KW, OH, OW), (
        strides[0], strides[1], strides[2], strides[3], strides[2] * SH, strides[3] * SW))

    col = col.transpose((0, 4, 5, 1, 2, 3)).reshape((N * OH * OW, -1))

    return col


def check_speed(x, function):
    t = time.time()
    col = function(x, (5,5), 1, 0)
    print(time.time() - t)
    return col


np.random.seed(0)
x = np.random.randint(0, 10, (200, 300, 10, 10))


check_speed(x, im2col_for_O)
check_speed(x, im2col_for_K)
check_speed(x, im2col_stride_O)
check_speed(x, im2col_stride_K)

# col1 = check_speed(x, im2col_for_O)
# col2 = check_speed(x, im2col_for_K)
# col3 = check_speed(x, im2col_stride_O)
# col4 = check_speed(x, im2col_stride_K)
# print(np.array_equal(col1, col2), np.array_equal(col2, col3), np.array_equal(col3, col4))
