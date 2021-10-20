import time
from Layer import *


class nn:
    def __init__(self, costfun):
        self.layers = []
        self.w = []
        self.b = []
        self.params = []
        self.costfun = costfun

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)

        return x

    def add(self, xlen, ylen, *layers):
        for layer in layers:
            if hasattr(layer, "init_weight"):
                self.params.extend(layer.init_weight(xlen, ylen))
        self.layers.extend(layers)

    def fit(self, x, t, batch_size, epochs, opti):
        a = time.time()
        if x.ndim == 1: x = x[np.newaxis].T
        if t.ndim == 1: t = t[np.newaxis].T
        iteration = t.shape[0] / batch_size
        for i in range(int(epochs * iteration)):
            batch_mask = np.random.choice(t.shape[0], batch_size, replace=False)
            x_batch = x[batch_mask]
            y_batch = t[batch_mask]
            costfun = lambda: self.costfun.forward(self.predict(x_batch), y_batch)
            grad = self.gradient(costfun)
            for e, param in enumerate(self.params):
                # grad = numerical_diff(param,costfun)
                opti(param, grad[e])

            if i % iteration == 0:
                print(f'\nepoch {int(i / iteration)} Total time {time.time() - a} fps {(time.time() - a) / (i + 1)}')
                print(f'cost {costfun()} accuracy {self.accuracy(x, t)}')

    def gradient(self, costfun):
        costfun()
        grad = []
        dout = self.costfun.backward()
        for layer in self.layers[::-1]:
            dout = layer.backward(dout)

        for layer in self.layers:
            if hasattr(layer, "init_weight"):
                grad.extend(layer.weightgrad())

        return grad

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1: t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
