import time

import numpy as np
import copy


class GD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, x, grad):
        x -= self.lr * grad


class weightopti:
    def init_weight(self, model):
        self.model = model
        self.v = {}
        for param in model.params:
            self.v[id(param)] = np.zeros_like(param)


class momentum(weightopti):
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum

    def update(self, x, grad):
        self.v[id(x)] = self.momentum * self.v[id(x)] - self.lr * grad
        x += self.v[id(x)]


class NAG(momentum):
    def update(self, x, grad):
        self.v[id(x)] = self.momentum * self.v[id(x)] - self.lr * grad
        x += self.momentum * self.v[id(x)] - self.lr * grad


class Adagrad(weightopti):
    def __init__(self, lr):
        self.lr = lr

    def update(self, x, grad):
        self.v[id(x)] += np.square(grad)
        x -= np.multiply(self.lr / (np.sqrt(self.v[id(x)] + 1e-7)),
                         grad)


class RMSProp(weightopti):
    def __init__(self, lr, RMSProp=0.9):
        self.lr = lr
        self.RMSProp = RMSProp

    def update(self, x, grad):
        self.v[id(x)] = self.RMSProp * self.v[id(x)] + (1 - self.RMSProp) * np.square(grad)
        x -= np.multiply(self.lr / (np.sqrt(self.v[id(x)] + 1e-7)),
                         grad)


class AdaDelta(weightopti):
    def __init__(self, AdaDelta=0.9):
        self.AdaDelta = AdaDelta

    def init_weight(self, model):
        super().init_weight(model)
        self.s = copy.deepcopy(self.v)

    def update(self, x, grad):
        self.v[id(x)] = self.AdaDelta * self.v[id(x)] + (1 - self.AdaDelta) * np.square(grad)
        d_t = np.multiply(np.sqrt(self.s[id(x)] + 1e-7) / np.sqrt(self.v[id(x)] + 1e-7),
                          grad)

        x -= d_t
        self.s[id(x)] = self.AdaDelta * self.s[id(x)] + (1 - self.AdaDelta) * np.square(d_t)


class Adam(weightopti):
    def __init__(self, lr, beta_1=0.9, beta_2=0.999):

        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def init_weight(self, model):
        super().init_weight(model)
        self.m = copy.deepcopy(self.v)
        self.mhat = copy.deepcopy(self.v)
        self.vhat = copy.deepcopy(self.v)

    def update(self, x, grad):
        self.m[id(x)] = self.beta_1 * self.m[id(x)] + (1 - self.beta_1) * grad
        self.mhat[id(x)] = self.m[id(x)] / (1 - self.beta_1 * self.beta_1)
        self.v[id(x)] = self.beta_2 * self.v[id(x)] + (1 - self.beta_2) * grad * grad
        self.vhat[id(x)] = self.v[id(x)] / (1 - self.beta_2 * self.beta_2)
        x -= self.lr * self.mhat[id(x)] / np.sqrt(self.vhat[id(x)] + 1e-7)
