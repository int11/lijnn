import numpy as np
import time


class nn:
    h = 1e-4  # 0.0001
    epsilon = 1e-7

    def __init__(self, x, y, costfun):
        self.x = x
        if self.x.ndim == 1: self.x = self.x[np.newaxis].T
        self.y = y
        if self.y.ndim == 1: self.y = self.y[np.newaxis].T
        self.train_size = self.y.shape[0]
        self.layers = []

        if costfun == "mse":
            def cost_fun():
                return np.sum((self.activation_fun() - self.y_batch) ** 2) / self.batch_size
        elif costfun == "binary_crossentropy":
            def cost_fun():
                return -np.sum(
                    self.y_batch * np.log(self.activation_fun() + self.epsilon) + (1 - self.y_batch) * np.log(
                        (1 - self.activation_fun()) + self.epsilon)) / self.batch_size
        elif costfun == "categorical_crossentropy":
            def cost_fun():
                return -np.sum(self.y_batch * np.log(self.activation_fun() + self.epsilon)) / self.batch_size
        else:
            raise
        self.cost_fun = cost_fun

    def activation_fun(self):
        result = self.x_batch
        for layer in self.layers:
            result = layer(result)

        return result

    def predict(self, x):
        result = x
        for layer in self.layers:
            result = layer(result)

        return np.round(result, 3)

    def add(self, layer):
        self.layers.append(layer)

    def fit(self, batch_size, epochs, opti):
        a = time.time()
        a1 = 0.
        self.batch_size = batch_size

        for i in range(epochs):
            batch_mask = np.random.choice(self.train_size, self.batch_size, replace=False)
            self.x_batch = self.x[batch_mask]
            self.y_batch = self.y[batch_mask]
            for layer in self.layers:
                opti(layer.w)
                opti(layer.b)

            if i % 100 == 0:
                a1 += 1
                print("time  ", (time.time() - a) / a1)
                cost = self.cost_fun()
                print(cost, sep='\n')

    def numerical_diff(self, x):
        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
        tmp = np.zeros_like(x)
        for i in it:
            idx = it.multi_index
            args = x[idx]
            x[idx] = args + self.h
            f1 = self.cost_fun()
            x[idx] = args - self.h
            f2 = self.cost_fun()
            x[idx] = args
            tmp[idx] = (f1 - f2) / (2 * self.h)
        return tmp

