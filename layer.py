import numpy as np


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
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.w) + self.b

        return out

    def backward(self, dout):
        print(dout)
        print(self.w.T)
        print(self.x.T)
        dx = np.dot(dout, self.w.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        dx = dx.reshape(*self.original_x_shape)  # 입력 데이터 모양 변경(텐서 대응)

        return dx


class Dropout:
    def __init__(self, probability):
        self.probability = probability

    def __call__(self, x):
        self.mask = np.random.rand(*x.shape) > self.probability
        return x * self.mask
