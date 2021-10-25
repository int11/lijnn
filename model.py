import time

import optimizer
from Layer import *


class nn:
    def __init__(self, costlayer, weight_decay_lambda=0):
        self.layers = []
        self.params = {}
        self.costlayer = costlayer
        self.weight_decay_lambda = weight_decay_lambda
        if self.weight_decay_lambda:
            def cost(x, t):
                weight_decay = 0
                for layer in self.layers:
                    if isinstance(layer, Dense):
                        w = layer.w
                        weight_decay += 0.5 * self.weight_decay_lambda * np.sum(w ** 2)
                return self.costlayer.forward(self.predict(x), t) + weight_decay
        else:
            cost = lambda x, t: self.costlayer.forward(self.predict(x), t)
        self.cost = cost

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def add(self, *layers):
        for layer in layers:
            if isinstance(layer, weightlayer):
                inputsize, outputsize = layer.getsize()
                if inputsize: self.inputsize = inputsize
                if outputsize: self.outputsize = outputsize
                print(layer.count, layer, self.inputsize, self.outputsize)
                layer.setsize(self.inputsize, self.outputsize)
                for key, value in layer.init_weight().items():
                    self.params[f'{key}{layer.count}'] = value

                self.inputsize = self.outputsize
        self.layers.extend(layers)

    def fit(self, x, t, batch_size, epochs, opti):

        if isinstance(opti, optimizer.weightopti): opti.init_weight(self)
        a = time.time()
        if x.ndim == 1: x = x[np.newaxis].T
        if t.ndim == 1: t = t[np.newaxis].T
        iteration = t.shape[0] / batch_size
        for i in range(int(epochs * iteration)):
            batch_mask = np.random.choice(t.shape[0], batch_size, replace=False)
            x_batch = x[batch_mask]
            t_batch = t[batch_mask]

            grad = self.gradient(x_batch, t_batch)
            for e, param in enumerate(self.params):
                # grad = numerical_diff(param,costfun)
                opti.update(param, grad[e])

            if i % iteration == 0:
                print(f'\nepoch {int(i / iteration)} Total time {time.time() - a} fps {(time.time() - a) / (i + 1)} '
                      f'\ncost {self.cost(x, t)} accuracy {self.accuracy(x, t)}')

    def gradient(self, x, t):
        self.cost(x, t)
        dout = self.costlayer.backward()
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

        grad = {}
        for layer in self.layers:
            if isinstance(layer, weightlayer):
                grad.update(layer.getgrad())

        return grad

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1: t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
