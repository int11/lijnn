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
        self.w = []
        self.b = []
        self.actifuns = []

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
        for w, b, actifun in zip(self.w, self.b, self.actifuns):
            result = actifun(result, w, b)

        return result

    def predict(self, x):
        result = x
        for w, b, actifun in zip(self.w, self.b, self.actifuns):
            result = actifun(result, w, b)

        return np.round(result, 3)

    def add(self, ylen, actifun, initialization=None):
        if self.w:
            input = self.w[-1].shape[1]
        else:
            input = self.x.shape[-1]

        if initialization == 'Xavier':
            m = np.sqrt(6 / (input + ylen))
            self.w.append(np.random.uniform(-m, m, (input, ylen)))
            self.b.append(np.random.uniform(-m, m, (1, ylen)))
        elif initialization == 'He':
            m = np.sqrt(6 / input)
            self.w.append(np.random.uniform(-m, m, (input, ylen)))
            self.b.append(np.random.uniform(-m, m, (1, ylen)))
        elif initialization is None:
            self.w.append(np.random.uniform(-1, 1, (input, ylen)))
            self.b.append(np.random.uniform(-1, 1, (1, ylen)))
        self.actifuns.append(actifun)

    def fit(self, batch_size, epochs, opti):
        a = time.time()
        a1 = 0.
        self.batch_size = batch_size

        for i in range(epochs):
            batch_mask = np.random.choice(self.train_size, self.batch_size, replace=False)
            self.x_batch = self.x[batch_mask]
            self.y_batch = self.y[batch_mask]
            for w, b in zip(self.w, self.b):
                opti(w)
                opti(b)

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
