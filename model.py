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
        layers = self.layers[::-1]
        for layer in layers:
            dout = layer.backward(dout)

        for layer in self.layers:
            grad.append(layer.dW)
            grad.append(layer.db)
        return grad

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1: t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
