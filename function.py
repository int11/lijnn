import numpy as np


class cost:
    @staticmethod
    def mse(predict, t, len):
        return np.sum((predict - t) ** 2) / len

    @staticmethod
    def binary_crossentropy(y, t, len):
        return -np.sum(t * np.log(y + 1e-7) + (1 - t) * np.log(
            (1 - y) + 1e-7)) / len



class init:
    Xavier = "Xavier"
    He = "He"


def oneshotencoding(data):
    return np.eye(np.max(data) + 1)[data]


def numerical_diff(w, f):
    it = np.nditer(w, flags=['multi_index'], op_flags=['readwrite'])
    tmp = np.zeros_like(w)
    for i in it:
        idx = it.multi_index
        args = w[idx]
        w[idx] = args + 1e-4
        f1 = f()

        w[idx] = args - 1e-4
        f2 = f()
        w[idx] = args
        tmp[idx] = (f1 - f2) / (2 * 1e-4)
    return tmp


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]