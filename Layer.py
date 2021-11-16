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
        return self.params


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
            m = np.sqrt(2 / (self.inputsize + self.outputsize))
        elif self.initialization == 'He':
            m = np.sqrt(2 / self.inputsize)
        else:
            m = self.initialization
        self.params['w'] = m * np.random.randn(self.inputsize, self.outputsize)
        self.params['b'] = m * np.random.randn(1, self.outputsize)
        return self.params

    def forward(self, x):
        self.original_x_shape = x.shape
        x.resize(x.shape[0], -1)
        self.x = x
        return np.dot(self.x, self.params['w']) + self.params['b']

    def backward(self, dout):
        # http://cs231n.stanford.edu/handouts/derivatives.pdf

        dx = np.dot(dout, self.params['w'].T)
        self.grad['w'] = np.dot(self.x.T, dout)
        self.grad['b'] = np.sum(dout, axis=0)

        return dx.reshape(*self.original_x_shape)


class BatchNormalization(weightlayer):
    # http://arxiv.org/abs/1502.03167
    def __init__(self, inputsize=None, outputsize=None):
        super().__init__(inputsize, outputsize)

    def init_weight(self, inputsize=None, outputsize=None):
        self.params['gamma'] = np.ones(self.outputsize)
        self.params['beta'] = np.zeros(self.outputsize)
        return self.params

    def forward(self, x):
        self.input_shape = x.shape
        if x.ndim != 2:
            x.resize(x.shape[0], -1)

        mu = x.mean(axis=0)
        self.xc = x - mu
        var = np.mean(self.xc ** 2, axis=0)
        self.std = np.sqrt(var + 10e-7)
        self.xn = self.xc / self.std

        self.batch_size = x.shape[0]
        out = self.params['gamma'] * self.xn + self.params['beta']

        return out.reshape(*self.input_shape)

    def backward(self, dout):
        if dout.ndim != 2:
            dout.resize(dout.shape[0], -1)

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

        return dx.reshape(*self.input_shape)


class Dropout:
    def __init__(self, probability=0.5):
        self.probability = probability

    def forward(self, x):
        self.mask = np.random.rand(*x.shape) > self.probability
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask


class Convolution(weightlayer):
    def __init__(self, inputsize=None, outputsize=None, stride=1, pad=0):
        super().__init__(inputsize, outputsize)
        self.stride = stride
        self.pad = pad

    def init_weight(self):
        return self.params

    def forward(self, x):
        FN, C, FH, FW = self.params['w'].shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2 * self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2 * self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)  # 1
        col_W = self.params['w'].reshape(FN, -1).T  # 2

        out = np.dot(col, col_W) + self.params['b']  # 3
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)  # 4

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.params['w'].shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)  # 4

        dW = np.dot(self.col.T, dout)  # 3
        self.grad['w'] = dW.transpose(1, 0).reshape(FN, C, FH, FW)  # 2
        self.grad['b'] = np.sum(dout, axis=0)  # 3

        dcol = np.dot(dout, self.col_W.T)  # 3
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)  # 1

        return dx


class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dcol = dmax.reshape(dout.shape[0] * dout.shape[1] * dout.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

        return dx
