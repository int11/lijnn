import numpy as np


class cost:
    @staticmethod
    def mse(predict, y, len):
        return np.sum((predict - y) ** 2) / len

    @staticmethod
    def binary_crossentropy(y, t, len):
        return -np.sum(t * np.log(y + 1e-7) + (1 - t) * np.log(
            (1 - y) + 1e-7)) / len



class initialization:
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
