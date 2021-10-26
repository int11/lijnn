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
                for key, value in self.params.items():
                    if key.count('w'):
                        weight_decay += 0.5 * self.weight_decay_lambda * np.sum(value ** 2)
                return self.costlayer.forward(self.predict(x), t) + weight_decay

            # weight_decay_backpropagation
            def deco(fun):
                def decofun(*args, **kwargs):
                    grad = fun(*args, **kwargs)
                    for key, value in grad.items():
                        if key.count('w'):
                            grad[key] += self.weight_decay_lambda * self.params[key]
                    return grad

                return decofun

            self.gradient = deco(self.gradient)
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
        if isinstance(opti, optimizer.weightopti): opti.init_weight(self.params)
        a = time.time()
        if x.ndim == 1: x = x[np.newaxis].T
        if t.ndim == 1: t = t[np.newaxis].T
        iteration = t.shape[0] / batch_size
        for i in range(int(epochs * iteration)):
            batch_mask = np.random.choice(t.shape[0], batch_size, replace=False)
            x_batch = x[batch_mask]
            t_batch = t[batch_mask]

            grads = self.gradient(x_batch, t_batch)
            # grads = self.grads_numerical(x_batch, t_batch)
            for param, grad in zip(self.params.values(), grads.values()):
                opti.update(param, grad)

            if i % iteration == 0:
                print(f'\nepoch {int(i / iteration)} Total time {time.time() - a} fps {(time.time() - a) / (i + 1)} '
                      f'\ncost {self.cost(x, t)} accuracy {self.accuracy(x, t)}')

    def grads_numerical(self, x, t):
        cost = lambda: self.cost(x, t)
        grad = {}
        for key, value in self.params.items():
            grad[key] = numerical_diff(value, cost)
        return grad

    def gradient(self, x, t):
        self.cost(x, t)
        dout = self.costlayer.backward()
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

        grad = {}
        for layer in self.layers:
            if isinstance(layer, weightlayer):
                for key, value in layer.getgrad().items():
                    grad[f'{key}{layer.count}'] = value

        return grad

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1: t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
