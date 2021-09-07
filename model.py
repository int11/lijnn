import time
from function import *

class nn:
    def __init__(self, costfun):
        self.layers = []
        self.w = []
        self.b = []
        self.costfun = costfun

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
        for i in range(epochs):
            batch_mask = np.random.choice(y.shape[0], batch_size, replace=False)
            x_batch = x[batch_mask]
            y_batch = y[batch_mask]
            costfun = lambda: self.costfun(self.predict(x_batch), y_batch, batch_size)
            for w, b in zip(self.w, self.b):
                grad = numerical_diff(w, costfun)
                opti(w, grad)
                grad = numerical_diff(b, costfun)
                opti(b, grad)

            a1 += 1
            print("time  ", (time.time() - a) / a1)
            print(costfun(), sep='\n')
            print(y_batch[:5])
            print(np.round(self.predict(x_batch)[:5],3))


