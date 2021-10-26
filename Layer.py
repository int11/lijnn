from function import *
from abc import *


class weightlayer(metaclass=ABCMeta):
    def __init_subclass__(cls):
        cls.count = 0

    def __init__(self, inputsize=None, outputsize=None):
        self.params, self.grad = {}, {}
        self.count = self.__class__.count
        self.__class__.count += 1

        self.inputsize, self.outputsize = inputsize, outputsize
        if not self.outputsize:
            self.outputsize, self.inputsize = self.inputsize, self.outputsize

    def setsize(self, inputsize, outputsize):
        self.inputsize, self.outputsize = inputsize, outputsize

    def getsize(self):
        return self.inputsize, self.outputsize

    def getgrad(self):
        return self.grad

    @abstractmethod
    def init_weight(self):
        pass


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        x[self.mask] = 0

        return x

    def backward(self, dout):
        dout[self.mask] = 0
        return dout


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


class Dense(weightlayer):
    def __init__(self, inputsize=None, outputsize=None, initialization=None):
        super().__init__(inputsize, outputsize)
        self.initialization = initialization

    def init_weight(self):
        if self.initialization == 'Xavier':
            m = np.sqrt(6 / (self.inputsize + self.outputsize))
            self.params['w'] = np.random.uniform(-m, m, (self.inputsize, self.outputsize))
            self.params['b'] = np.random.uniform(-m, m, (1, self.outputsize))
        elif self.initialization == 'He':
            m = np.sqrt(6 / self.inputsize)
            self.params['w'] = np.random.uniform(-m, m, (self.inputsize, self.outputsize))
            self.params['b'] = np.random.uniform(-m, m, (1, self.outputsize))
        elif self.initialization is None:
            self.params['w'] = np.random.uniform(-1, 1, (self.inputsize, self.outputsize))
            self.params['b'] = np.random.uniform(-1, 1, (1, self.outputsize))
        return self.params

    def forward(self, x):
        self.x = x
        return np.dot(self.x, self.params['w']) + self.params['b']

    def backward(self, dout):
        # http://cs231n.stanford.edu/handouts/derivatives.pdf

        dx = np.dot(dout, self.params['w'].T)
        self.grad['w'] = np.dot(self.x.T, dout)
        self.grad['b'] = np.sum(dout, axis=0)

        return dx


class BatchNormalization(weightlayer):
    # http://arxiv.org/abs/1502.03167
    def __init__(self, inputsize=None, outputsize=None):
        super().__init__(inputsize, outputsize)

    def init_weight(self, inputsize=None, outputsize=None):
        self.params['gamma'] = np.ones(self.outputsize)
        self.params['beta'] = np.zeros(self.outputsize)
        return self.params

    def forward(self, x):
        mu = x.mean(axis=0)
        self.xc = x - mu
        var = np.mean(self.xc ** 2, axis=0)
        self.std = np.sqrt(var + 10e-7)
        self.xn = self.xc / self.std

        self.batch_size = x.shape[0]

        return self.params['gamma'] * self.xn + self.params['beta']

    def backward(self, dout):
        self.grad['beta'] = dout.sum(axis=0)
        self.grad['gamma'] = np.sum(self.xn * dout, axis=0)
        dxn = self.params['gamma'] * dout
        # f = x/y
        # df/dx = 1/y
        # dL/df * df/dx = dxn * 1/y
        # dL/dx = dxn/y
        dxc = dxn / self.std
        # df/dy = -x/y^2
        # dL/df * df/dx = dxn * -x/y^2
        # dL/dx = -x * dxn / y^2
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        # dvar = 0.5 * 1. / np.sqrt(var + 10e-7) * dstd
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
    def __init__(self, probability=0.5):
        self.probability = probability

    def forward(self, x):
        self.mask = np.random.rand(*x.shape) > self.probability
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask
