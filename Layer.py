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
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx


class Softmax:
    def forward(self, x):
        softmax = np.exp(x - (x.max(axis=1).reshape([-1, 1])))
        softmax /= softmax.sum(axis=1).reshape([-1, 1])
        self.out = softmax
        return self.out

    def backward(self, dout):
        return (1 + dout) * self.out / dout.shape[0]


class categorical_crossentropy:
    def forward(self, y, t):
        self.predict = y
        self.y = t
        self.batch_size = t.shape[0]
        self.out = -np.sum(t * np.log(y + 1e-7)) / self.batch_size

        return self.out

    def backward(self, dout=1):
        return -(self.y / self.predict)


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
        if not isinstance(actilayer, list):
            actilayer = [actilayer]
        self.actilayer = actilayer

    def forward(self, x):
        self.x = x
        out = np.dot(self.x, self.w) + self.b
        for i in self.actilayer:
            out = i.forward(out)
        return out

    def backward(self, dout):
        # http://cs231n.stanford.edu/handouts/derivatives.pdf

        for i in self.actilayer[::-1]:
            dout = i.backward(dout)
        dx = np.dot(dout, self.w.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx


class BatchNormalization:
    # http://arxiv.org/abs/1502.03167

    def __init__(self, gamma, beta):
        self.gamma = gamma
        self.beta = beta

    def forward(self, x):
        mu = x.mean(axis=0)
        self.xc = x - mu
        var = np.mean(self.xc ** 2, axis=0)
        self.std = np.sqrt(var + 10e-7)
        self.xn = self.xc / self.std

        self.batch_size = x.shape[0]

        return self.gamma * self.xn + self.beta

    def backward(self, dout):
        self.dbeta = dout.sum(axis=0)
        self.dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        # f = x/y
        # df/dx = 1/y
        # dL/df * df/dx = dxn * 1/y
        # dL/dx = dxn/y
        dxc = dxn / self.std
        # df/dy = -x/y^2
        # dL/df * df/dx = dxn * -x/y^2
        # dL/dx = -x * dxn / y^2
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        # dvar = 0.5 * 1. / np.sqrt(self.std + 10e-7) * dstd
        # np.sqrt(var + 10e-7) = self.std
        # dvar = 0.5 * 1. / self.std * dstd
        # dvar = 0.5 * dstd / self.std
        dvar = 0.5 * dstd / self.std

        dxc += 2.0 * self.xc * dvar / self.batch_size
        # f = x - y
        # df/dx = 1
        # df/dy = -1
        dx = dxc - np.sum(dxc, axis=0) / self.batch_size

        return dx


class Dropout:
    def __init__(self, probability):
        self.probability = probability

    def forward(self, x):
        self.mask = np.random.rand(*x.shape) > self.probability
        return x * self.mask

    def backward(self):
        return
