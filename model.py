import time
from function import *
import Layer


class nn:
    def __init__(self, costfun):
        self.layers = []
        self.w = []
        self.b = []
        self.params = []
        self.costfun = costfun

    def predict(self, x):
        result = x
        for layer in self.layers:
            result = layer.forward(result)

        return result

    def add(self, layer):
        self.layers.append(layer)
        self.w.append(layer.w)
        self.b.append(layer.b)
        self.params.append(layer.w)
        self.params.append(layer.b)

    def fit(self, x, y, batch_size, epochs, opti):
        a = time.time()
        a1 = 0.
        if x.ndim == 1: x = x[np.newaxis].T
        if y.ndim == 1: y = y[np.newaxis].T
        for i in range(epochs):
            batch_mask = np.random.choice(y.shape[0], batch_size, replace=False)
            x_batch = x[batch_mask]
            y_batch = y[batch_mask]
            costfun = lambda: self.costfun.forward(self.predict(x_batch), y_batch)
            self.gradient(costfun())
            for param in self.params:
                opti(param, numerical_diff(param, costfun))

            a1 += 1
            print("time  ", (time.time() - a) / a1)
            print(costfun(), sep='\n')
            print(y_batch[:5])
            print(np.round(self.predict(x_batch)[:5], 3))

    def gradient(self, cost):
        dout = self.costfun.backward()
        layers = self.layers[::-1]
        for layer in layers:
            dout = layer.backward(dout)
            print(dout)
