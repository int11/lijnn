import numpy as np


class nn:
    def __init__(self, x, y, actifuns, costfuns, xlen, ylen):
        self.h = 1e-4  # 0.0001
        self.delta = 1e-7
        self.learning_rate = 1e-2
        self.x = x
        self.y = oneshotencoding(y)
        if self.x.ndim == 1:
            self.x = self.x[np.newaxis]
        self.lensample = len(self.x[0])
        self.w = np.zeros((xlen, ylen))
        self.b = np.zeros((1, ylen))

        if actifuns == "linear":
            def activation_fun():
                return np.matmul(self.x, self.w) + self.b
        elif actifuns == "sigmoid":
            def activation_fun():
                return 1 / (1 + np.exp(-(np.matmul(self.x, self.w) + self.b)))
        elif actifuns == "softmax":
            def activation_fun():
                x_data = np.matmul(self.x, self.w) + self.b
                softmax = np.exp(x_data - (x_data.max(axis=1).reshape([-1, 1])))
                softmax /= softmax.sum(axis=1).reshape([-1, 1])
                return softmax
        else:
            raise
        self.activation_fun = activation_fun

        if costfuns == "mse":
            def cost_fun():
                return np.sum((self.activation_fun() - self.y) ** 2) / self.lensample
        elif costfuns == "binary_crossentropy":
            def cost_fun():
                return -np.sum(self.y * np.log(self.activation_fun() + self.delta) + (1 - self.y) * np.log(
                    (1 - self.activation_fun()) + self.delta))
        elif costfuns == "categorical_crossentropy":
            def cost_fun():
                return -np.sum(self.y * np.log(self.activation_fun() + self.delta)) / self.lensample
        else:
            raise
        self.cost_fun = cost_fun

    def __call__(self):
        self.numerical_diff(self.cost_fun, self.w)
        self.numerical_diff(self.cost_fun, self.b)

    def numerical_diff(self, f, x):
        for i in range(x.size):
            args = x.reshape(-1)[i]
            x.reshape(-1)[i] = args + self.h
            f1 = f()
            x.reshape(-1)[i] = args - self.h
            f2 = f()
            x.reshape(-1)[i] = args
            x.reshape(-1)[i] -= self.learning_rate * (f1 - f2) / (2 * self.h)


def oneshotencoding(data):
    return np.eye(np.max(data) + 1)[data]


x = np.array(
    [[5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2], [5.8, 2.6, 4.0, 1.2], [6.7, 3.0, 5.2, 2.3], [5.6, 2.8, 4.9, 2.0]])
y = np.array([0, 0, 1, 2, 2])
nn = nn(x, y, "softmax", "categorical_crossentropy", 4, 3)


for i in range(10000):
    nn()
    if i%100==0:
        print(nn.cost_fun(), nn.w, nn.b, sep='\n')


print(np.round(nn.activation_fun(),3))