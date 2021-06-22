import numpy as np


class activation:

    @staticmethod
    def linear(x, w, b):
        return np.matmul(x, w) + b

    @staticmethod
    def sigmoid(x, w, b):
        return 1 / (1 + np.exp(-(np.matmul(x, w) + b)))

    @staticmethod
    def softmax(x, w, b):
        x1 = np.matmul(x, w) + b
        softmax = np.exp(x1 - (x1.max(axis=1).reshape([-1, 1])))
        softmax /= softmax.sum(axis=1).reshape([-1, 1])
        return softmax

    @staticmethod
    def relu(x, w, b):
        x1 = np.matmul(x, w) + b
        return np.maximum(0, x1)

    @staticmethod
    def leaky_relu(x, w, b):
        x1 = np.matmul(x, w) + b
        return np.maximum(0.1 * x1, x1)


class cost:
    mse = "mse"
    binary_crossentropy = "binary_crossentropy"
    categorical_crossentropy = "categorical_crossentropy"


class nn:
    def __init__(self, x, y, xlen, ylen, actifun, costfuns, batch_size, learning_rate=0.01):
        self.h = 1e-4  # 0.0001
        self.delta = 1e-7
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.x = x
        self.y = y
        if self.x.ndim == 1:
            self.x = self.x[np.newaxis]
        self.train_size = self.y.shape[0]
        self.w = [np.ones((xlen, ylen))]
        self.b = [np.ones((1, ylen))]
        self.actifunlist = [actifun]

        if costfuns == "mse":
            def cost_fun():
                return np.sum((self.activation_fun() - self.y_batch) ** 2) / self.batch_size
        elif costfuns == "binary_crossentropy":
            def cost_fun():
                return -np.sum(self.y_batch * np.log(self.activation_fun() + self.delta) + (1 - self.y) * np.log(
                    (1 - self.activation_fun()) + self.delta)) / self.batch_size
        elif costfuns == "categorical_crossentropy":
            def cost_fun():
                return -np.sum(self.y_batch * np.log(self.activation_fun() + self.delta)) / self.batch_size
        else:
            raise
        self.cost_fun = cost_fun

    def activation_fun(self):
        result = self.x_batch
        for w, b, actifun in zip(self.w, self.b, self.actifunlist):
            result = actifun(result, w, b)

        return result

    def predict(self):
        result = self.x
        for w, b, actifun in zip(self.w, self.b, self.actifunlist):
            result = actifun(result, w, b)

        return result

    def add(self, ylen, actifun):
        self.w.append(np.ones((self.w[-1].shape[1], ylen)))
        self.b.append(np.ones((1, ylen)))
        self.actifunlist.append(actifun)

    def __call__(self):
        batch_mask = np.random.choice(self.train_size,self.batch_size)
        self.x_batch = self.x[batch_mask]
        self.y_batch = self.y[batch_mask]
        for w, b in zip(self.w, self.b):
            self.numerical_diff(w)
            self.numerical_diff(b)

    def numerical_diff(self, x):
        for i in range(x.size):
            args = x.flat[i]
            x.flat[i] = args + self.h
            f1 = self.cost_fun()
            x.flat[i] = args - self.h
            f2 = self.cost_fun()
            x.flat[i] = args
            x.flat[i] -= self.learning_rate * (f1 - f2) / (2 * self.h)


def oneshotencoding(data):
    return np.eye(np.max(data) + 1)[data]


x = np.array(
    [[5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2], [4.7, 3.2, 1.3, 0.2], [4.6, 3.1, 1.5, 0.2], [5.0, 3.6, 1.4, 0.2],
     [5.4, 3.9, 1.7, 0.4], [4.6, 3.4, 1.4, 0.3], [5.0, 3.4, 1.5, 0.2], [4.4, 2.9, 1.4, 0.2], [4.9, 3.1, 1.5, 0.1],
     [5.4, 3.7, 1.5, 0.2], [4.8, 3.4, 1.6, 0.2], [4.8, 3.0, 1.4, 0.1], [4.3, 3.0, 1.1, 0.1], [5.8, 4.0, 1.2, 0.2],
     [5.7, 4.4, 1.5, 0.4], [5.4, 3.9, 1.3, 0.4], [5.1, 3.5, 1.4, 0.3], [5.7, 3.8, 1.7, 0.3], [5.1, 3.8, 1.5, 0.3],
     [5.4, 3.4, 1.7, 0.2], [5.1, 3.7, 1.5, 0.4], [4.6, 3.6, 1.0, 0.2], [5.1, 3.3, 1.7, 0.5], [4.8, 3.4, 1.9, 0.2],
     [5.0, 3.0, 1.6, 0.2], [5.0, 3.4, 1.6, 0.4], [5.2, 3.5, 1.5, 0.2], [5.2, 3.4, 1.4, 0.2], [4.7, 3.2, 1.6, 0.2],
     [4.8, 3.1, 1.6, 0.2], [5.4, 3.4, 1.5, 0.4], [5.2, 4.1, 1.5, 0.1], [5.5, 4.2, 1.4, 0.2], [4.9, 3.1, 1.5, 0.1],
     [5.0, 3.2, 1.2, 0.2], [5.5, 3.5, 1.3, 0.2], [4.9, 3.1, 1.5, 0.1], [4.4, 3.0, 1.3, 0.2], [5.1, 3.4, 1.5, 0.2],
     [5.0, 3.5, 1.3, 0.3], [4.5, 2.3, 1.3, 0.3], [4.4, 3.2, 1.3, 0.2], [5.0, 3.5, 1.6, 0.6], [5.1, 3.8, 1.9, 0.4],
     [4.8, 3.0, 1.4, 0.3], [5.1, 3.8, 1.6, 0.2], [4.6, 3.2, 1.4, 0.2], [5.3, 3.7, 1.5, 0.2], [5.0, 3.3, 1.4, 0.2],
     [7.0, 3.2, 4.7, 1.4], [6.4, 3.2, 4.5, 1.5], [6.9, 3.1, 4.9, 1.5], [5.5, 2.3, 4.0, 1.3], [6.5, 2.8, 4.6, 1.5],
     [5.7, 2.8, 4.5, 1.3], [6.3, 3.3, 4.7, 1.6], [4.9, 2.4, 3.3, 1.0], [6.6, 2.9, 4.6, 1.3], [5.2, 2.7, 3.9, 1.4],
     [5.0, 2.0, 3.5, 1.0], [5.9, 3.0, 4.2, 1.5], [6.0, 2.2, 4.0, 1.0], [6.1, 2.9, 4.7, 1.4], [5.6, 2.9, 3.6, 1.3],
     [6.7, 3.1, 4.4, 1.4], [5.6, 3.0, 4.5, 1.5], [5.8, 2.7, 4.1, 1.0], [6.2, 2.2, 4.5, 1.5], [5.6, 2.5, 3.9, 1.1],
     [5.9, 3.2, 4.8, 1.8], [6.1, 2.8, 4.0, 1.3], [6.3, 2.5, 4.9, 1.5], [6.1, 2.8, 4.7, 1.2], [6.4, 2.9, 4.3, 1.3],
     [6.6, 3.0, 4.4, 1.4], [6.8, 2.8, 4.8, 1.4], [6.7, 3.0, 5.0, 1.7], [6.0, 2.9, 4.5, 1.5], [5.7, 2.6, 3.5, 1.0],
     [5.5, 2.4, 3.8, 1.1], [5.5, 2.4, 3.7, 1.0], [5.8, 2.7, 3.9, 1.2], [6.0, 2.7, 5.1, 1.6], [5.4, 3.0, 4.5, 1.5],
     [6.0, 3.4, 4.5, 1.6], [6.7, 3.1, 4.7, 1.5], [6.3, 2.3, 4.4, 1.3], [5.6, 3.0, 4.1, 1.3], [5.5, 2.5, 4.0, 1.3],
     [5.5, 2.6, 4.4, 1.2], [6.1, 3.0, 4.6, 1.4], [5.8, 2.6, 4.0, 1.2], [5.0, 2.3, 3.3, 1.0], [5.6, 2.7, 4.2, 1.3],
     [5.7, 3.0, 4.2, 1.2], [5.7, 2.9, 4.2, 1.3], [6.2, 2.9, 4.3, 1.3], [5.1, 2.5, 3.0, 1.1], [5.7, 2.8, 4.1, 1.3],
     [6.3, 3.3, 6.0, 2.5], [5.8, 2.7, 5.1, 1.9], [7.1, 3.0, 5.9, 2.1], [6.3, 2.9, 5.6, 1.8], [6.5, 3.0, 5.8, 2.2],
     [7.6, 3.0, 6.6, 2.1], [4.9, 2.5, 4.5, 1.7], [7.3, 2.9, 6.3, 1.8], [6.7, 2.5, 5.8, 1.8], [7.2, 3.6, 6.1, 2.5],
     [6.5, 3.2, 5.1, 2.0], [6.4, 2.7, 5.3, 1.9], [6.8, 3.0, 5.5, 2.1], [5.7, 2.5, 5.0, 2.0], [5.8, 2.8, 5.1, 2.4],
     [6.4, 3.2, 5.3, 2.3], [6.5, 3.0, 5.5, 1.8], [7.7, 3.8, 6.7, 2.2], [7.7, 2.6, 6.9, 2.3], [6.0, 2.2, 5.0, 1.5],
     [6.9, 3.2, 5.7, 2.3], [5.6, 2.8, 4.9, 2.0], [7.7, 2.8, 6.7, 2.0], [6.3, 2.7, 4.9, 1.8], [6.7, 3.3, 5.7, 2.1],
     [7.2, 3.2, 6.0, 1.8], [6.2, 2.8, 4.8, 1.8], [6.1, 3.0, 4.9, 1.8], [6.4, 2.8, 5.6, 2.1], [7.2, 3.0, 5.8, 1.6],
     [7.4, 2.8, 6.1, 1.9], [7.9, 3.8, 6.4, 2.0], [6.4, 2.8, 5.6, 2.2], [6.3, 2.8, 5.1, 1.5], [6.1, 2.6, 5.6, 1.4],
     [7.7, 3.0, 6.1, 2.3], [6.3, 3.4, 5.6, 2.4], [6.4, 3.1, 5.5, 1.8], [6.0, 3.0, 4.8, 1.8], [6.9, 3.1, 5.4, 2.1],
     [6.7, 3.1, 5.6, 2.4], [6.9, 3.1, 5.1, 2.3], [5.8, 2.7, 5.1, 1.9], [6.8, 3.2, 5.9, 2.3], [6.7, 3.3, 5.7, 2.5],
     [6.7, 3.0, 5.2, 2.3], [6.3, 2.5, 5.0, 1.9], [6.5, 3.0, 5.2, 2.0], [6.2, 3.4, 5.4, 2.3], [5.9, 3.0, 5.1, 1.8]])
y = np.array(
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
nn = nn(x, oneshotencoding(y), 4, 3, activation.relu, cost.categorical_crossentropy, len(x))
nn.add(3, activation.relu)
nn.add(3, activation.softmax)
for i in range(10000):
    nn()
    if i % 100 == 0:
        cost = nn.cost_fun()
        print(cost, sep='\n')

print(np.round(nn.predict(), 3))
