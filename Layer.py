import numpy as np

from function import *


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
    def forward(self, x):
        self.out = activation.sigmoid(x)
        return self.out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx


class Softmax:
    def forward(self, x):
        self.out = activation.softmax(x)
        return self.out

    def backward(self, dout):
        return (1 + dout) * self.out

class categorical_crossentropy:
    def forward(self, predict, y):
        self.predict = predict
        self.y = y
        self.batch_size = y.shape[0]
        self.out = -np.sum(y * np.log(predict + 1e-7)) / self.batch_size
        return self.out

    def backward(self, dout=1):
        return (self.y/self.predict) * np.sqrt(self.out)


class Dense:
    def __init__(self, xlen, ylen, actilayer, initialization=None):
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
        self.actilayer = actilayer

    def forward(self, x):
        self.x = x
        out = np.dot(self.x, self.w) + self.b
        return self.actilayer.forward(out)

    def backward(self, dout):
        dout = self.actilayer.backward(dout)
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
