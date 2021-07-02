import numpy as np


class GD:
    def __init__(self, model, learning_rate=0.01):
        self.model = model
        self.learning_rate = learning_rate

    def __call__(self, x):
        x -= self.learning_rate * self.model.numerical_diff(x)


class momentum:
    def __init__(self, model, learning_rate=0.01, momentum=0.9):
        self.model = model
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.v = {}
        for w, b in zip(model.w, model.b):
            self.v[id(w)] = np.zeros_like(w)
            self.v[id(b)] = np.zeros_like(b)

    def __call__(self, x):
        self.v[id(x)] = self.momentum * self.v[id(x)] - self.learning_rate * self.model.numerical_diff(x)
        x += self.v[id(x)]

class NAG:
    def __init__(self, model, learning_rate=0.01, momentum=0.9):
        self.model = model
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.v = {}
        for w, b in zip(model.w, model.b):
            self.v[id(w)] = np.zeros_like(w)
            self.v[id(b)] = np.zeros_like(b)

    def __call__(self, x):
        print(self.model.numerical_diff(x+self.momentum*self.v[id(x)]))
        self.v[id(x)] = self.momentum * self.v[id(x)] - self.learning_rate * self.model.numerical_diff(x+self.momentum*self.v[id(x)])
        x += self.v[id(x)]
