import numpy as np


class activation:

    @staticmethod
    def linear(x, w, b):
        return np.matmul(x, w) + b

    @staticmethod
    def sigmoid(x, w, b):
        return 1 / (1 + np.exp(-(np.matmul(x, w) + b)))

    @staticmethod
    def softmax(x, w, b):
        x1 = np.matmul(x, w) + b
        softmax = np.exp(x1 - (x1.max(axis=1).reshape([-1, 1])))
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
    mse = "mse"
    binary_crossentropy = "binary_crossentropy"
    categorical_crossentropy = "categorical_crossentropy"


class initialization:
    Xavier = "Xavier"
    He = "He"


def oneshotencoding(data):
    return np.eye(np.max(data) + 1)[data]
