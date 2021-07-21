import numpy as np


class GD:
    def __init__(self, model, learning_rate=0.01):
        self.model = model
        self.learning_rate = learning_rate

    def __call__(self, x):
        x -= self.learning_rate * self.model.numerical_diff(x)


class idpaste:
    def __init__(self, model, learning_rate=0.01):
        self.model = model
        self.learning_rate = learning_rate
        self.v = {}
        for w, b in zip(model.w, model.b):
            self.v[id(w)] = np.zeros_like(w)
            self.v[id(b)] = np.zeros_like(b)


class momentum(idpaste):
    def __init__(self, model, learning_rate=0.01, momentum=0.9):
        super().__init__(model, learning_rate)
        self.momentum = momentum

    def __call__(self, x):
        self.v[id(x)] = self.momentum * self.v[id(x)] - self.learning_rate * self.model.numerical_diff(x)
        x += self.v[id(x)]


class NAG(momentum):
    def __call__(self, x):
        self.v[id(x)] = self.momentum * self.v[id(x)] - self.learning_rate * self.model.numerical_diff(x)
        x += self.momentum * self.v[id(x)] - self.learning_rate * self.model.numerical_diff(x)


class Adagrad(idpaste):
    def __call__(self, x):
        self.v[id(x)] += np.square(self.model.numerical_diff(x))
        x -= np.multiply(self.learning_rate / (np.sqrt(self.v[id(x)] + self.model.delta)), self.model.numerical_diff(x))


class RMSProp(idpaste):
    def __init__(self, model, learning_rate=0.01, RMSProp=0.9):
        super().__init__(model, learning_rate)
        self.RMSProp = RMSProp

    def __call__(self, x):
        self.v[id(x)] = self.RMSProp * self.v[id(x)] + (1-self.RMSProp) * np.square(self.model.numerical_diff(x))
        x -= np.multiply(self.learning_rate / (np.sqrt(self.v[id(x)] + self.model.delta)), self.model.numerical_diff(x))
