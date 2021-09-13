import numpy as np

import function


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        x[self.mask] = 0

        return x

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx

class softmax:
    def forward(self,x):
        softmax = np.exp(x - (x.max(axis=1).reshape([-1, 1])))
        softmax /= softmax.sum(axis=1).reshape([-1, 1])
        return softmax

    def backward(self, dout):
        return (1-dout) *

class Dense:
    def __init__(self, xlen, ylen, actifun, initialization=None):
        if initialization == 'Xavier':
            m = np.sqrt(6 / (xlen + ylen))
            self.w = np.random.uniform(-m, m, (xlen, ylen))
            self.b = np.random.uniform(-m, m, (1, ylen))
        elif initialization == 'He':
            m = np.sqrt(6 / xlen)
            self.w = np.random.uniform(-m, m, (xlen, ylen))
            self.b = np.random.uniform(-m, m, (1, ylen))
        elif initialization is None:
            self.w = np.random.uniform(-1, 1, (xlen, ylen))
            self.b = np.random.uniform(-1, 1, (1, ylen))
        self.actifun = actifun

    def forward(self, x):
        self.x = x
        out = np.dot(self.x, self.w) + self.b
        out = self.actifun.forward(out)
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.w.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx


class Dropout:
    def __init__(self, probability):
        self.probability = probability

    def __call__(self, x):
        self.mask = np.random.rand(*x.shape) > self.probability
        return x * self.mask
