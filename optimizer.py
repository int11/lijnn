import numpy as np
import copy


class GD:
    def __init__(self, model, lr):
        self.model = model
        self.lr = lr

    def __call__(self, x):
        x -= self.lr * self.model.numerical_diff(x)


class _idpaste:
    def __init__(self, model):
        self.model = model
        self.v = {}
        for w, b in zip(model.w, model.b):
            self.v[id(w)] = np.zeros_like(w)
            self.v[id(b)] = np.zeros_like(b)


class momentum(_idpaste):
    def __init__(self, model, lr=0.01, momentum=0.9):
        super().__init__(model)
        self.lr = lr
        self.momentum = momentum

    def __call__(self, x):
        self.v[id(x)] = self.momentum * self.v[id(x)] - self.lr * self.model.numerical_diff(x)
        x += self.v[id(x)]


class NAG(momentum):
    def __call__(self, x):
        self.v[id(x)] = self.momentum * self.v[id(x)] - self.lr * self.model.numerical_diff(x)
        x += self.momentum * self.v[id(x)] - self.lr * self.model.numerical_diff(x)


class Adagrad(_idpaste):
    def __init__(self, model, lr, epsilon=1e-7):
        super().__init__(model)
        self.lr = lr
        self.epsilon = epsilon

    def __call__(self, x):
        self.v[id(x)] += np.square(self.model.numerical_diff(x))
        x -= np.multiply(self.lr / (np.sqrt(self.v[id(x)] + self.epsilon)),
                         self.model.numerical_diff(x))


class RMSProp(_idpaste):
    def __init__(self, model, lr, RMSProp=0.9, epsilon=1e-7):
        super().__init__(model)
        self.lr = lr
        self.epsilon = epsilon
        self.RMSProp = RMSProp

    def __call__(self, x):
        self.v[id(x)] = self.RMSProp * self.v[id(x)] + (1 - self.RMSProp) * np.square(self.model.numerical_diff(x))
        x -= np.multiply(self.lr / (np.sqrt(self.v[id(x)] + self.epsilon)),
                         self.model.numerical_diff(x))


class AdaDelta(_idpaste):
    def __init__(self, model, AdaDelta=0.9, epsilon=1e-7):
        super().__init__(model)
        self.AdaDelta = AdaDelta
        self.epsilon = epsilon
        self.s = copy.deepcopy(self.v)

    def __call__(self, x):
        self.v[id(x)] = self.AdaDelta * self.v[id(x)] + (1 - self.AdaDelta) * np.square(self.model.numerical_diff(x))
        d_t = np.multiply(np.sqrt(self.s[id(x)] + self.epsilon) / np.sqrt(self.v[id(x)] + self.epsilon),
                          self.model.numerical_diff(x))

        x -= d_t
        self.s[id(x)] = self.AdaDelta * self.s[id(x)] + (1 - self.AdaDelta) * np.square(d_t)


class Adam(_idpaste):
    def __init__(self, model, lr, beta_1=0.9, beta_2=0.999, epsilon=1e-7):
        super().__init__(model)
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.m = copy.deepcopy(self.v)
        self.mhat = copy.deepcopy(self.v)
        self.vhat = copy.deepcopy(self.v)

    def __call__(self, x):
        self.m[id(x)] = self.beta_1 * self.m[id(x)] + (1 - self.beta_1) * self.model.numerical_diff(x)
        self.mhat[id(x)] = self.m[id(x)] / (1 - self.beta_1 * self.beta_1)
        self.v[id(x)] = self.beta_2 * self.v[id(x)] + (1 - self.beta_2) * self.model.numerical_diff(
            x) * self.model.numerical_diff(x)
        self.vhat[id(x)] = self.v[id(x)] / (1 - self.beta_2 * self.beta_2)
        x -= self.lr * self.mhat[id(x)] / np.sqrt(self.vhat[id(x)] + self.epsilon)
