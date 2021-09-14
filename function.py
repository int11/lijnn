import numpy as np


class activation:

    @staticmethod
    def linear(x, w, b):
        return np.matmul(x, w) + b

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def softmax(x):
        softmax = np.exp(x - (x.max(axis=1).reshape([-1, 1])))
        softmax /= softmax.sum(axis=1).reshape([-1, 1])
        return softmax

    @staticmethod
    def relu(x, w, b):
        x1 = np.matmul(x, w) + b
        return np.maximum(0, x1)

    @staticmethod
    def leaky_relu(x, w, b):
        x1 = np.matmul(x, w) + b
        return np.maximum(0.1 * x1, x1)


class cost:
    @staticmethod
    def mse(predict, y, len):
        return np.sum((predict - y) ** 2) / len

    @staticmethod
    def binary_crossentropy(predict, y, len):
        return -np.sum(y * np.log(predict + 1e-7) + (1 - y) * np.log(
            (1 - predict) + 1e-7)) / len

    @staticmethod
    def categorical_crossentropy(predict, y, len):
        return -np.sum(y * np.log(predict + 1e-7)) / len


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
