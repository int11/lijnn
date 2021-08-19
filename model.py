import numpy as np
import time


class nn:
    def __init__(self, costfun):
        self.layers = []
        self.w = []
        self.b = []

        if costfun == "mse":
            def cost_fun(x, y):
                return np.sum((self.predict(x) - y) ** 2) / self.batch_size
        elif costfun == "binary_crossentropy":
            def cost_fun(x, y):
                return -np.sum(y * np.log(self.predict(x) + 1e-7) + (1 - y) * np.log(
                    (1 - self.predict(x)) + 1e-7)) / self.batch_size
        elif costfun == "categorical_crossentropy":
            def cost_fun(x, y):
                return -np.sum(y * np.log(self.predict(x) + 1e-7)) / self.batch_size
        else:
            raise
        self.cost_fun = cost_fun

    def predict(self, x):
        result = x
        for layer in self.layers:
            result = layer(result)

        return result

    def add(self, layer):
        self.layers.append(layer)
        self.w.append(layer.w)
        self.b.append(layer.b)

    def fit(self, x, y, batch_size, epochs, opti):
        a = time.time()
        a1 = 0.
        if x.ndim == 1: x = x[np.newaxis].T
        if y.ndim == 1: y = y[np.newaxis].T
        self.batch_size = batch_size
        for i in range(epochs):
            batch_mask = np.random.choice(y.shape[0], self.batch_size, replace=False)
            x_batch = x[batch_mask]
            y_batch = y[batch_mask]
            for w, b in zip(self.w, self.b):
                a = time.time()
                grad = self.numerical_diff(x)
                print('grad', time.time() - a)
                opti(w, grad)
                opti(b, grad)

            a1 += 1
            print("time  ", (time.time() - a) / a1)
            cost = self.cost_fun(x_batch, y_batch)
            print(cost, sep='\n')

    def numerical_diff(self, w, f):
        it = np.nditer(w, flags=['multi_index'], op_flags=['readwrite'])
        tmp = np.zeros_like(w)
        for i in it:
            idx = it.multi_index
            args = w[idx]
            w[idx] = args + 1e-4
            f1 = self.cost_fun()
            w[idx] = args - 1e-4
            f2 = self.cost_fun()
            w[idx] = args
            tmp[idx] = (f1 - f2) / (2 * 1e-4)
        return tmp
