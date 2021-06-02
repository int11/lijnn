import numpy as np


class nn:
    def __init__(self, x, y):
        self.h = 1e-4  # 0.0001
        self.learning_rate = 1e-2
        self.x = x
        self.y = y
        if len(self.x.shape) == 1:
            self.x = self.x[np.newaxis]
        self.lensample = len(self.x[0])
        self.w = np.zeros(len(self.x))
        self.b = np.zeros(1)

    def __call__(self):

        def f():
            hypothesis = np.matmul(self.w, self.x) + self.b
            return self.cross_entropy(self.sigmoid(hypothesis))

        self.numerical_diff(f, self.w)
        self.numerical_diff(f, self.b)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def MSE(self, hypothesis):
        return np.sum((hypothesis - self.y) ** 2) / self.lensample

    def cross_entropy(self, hypothesis):
        delta = 1e-7
        return -np.sum(self.y * np.log(hypothesis + delta) + (1 - self.y) * np.log((1 - hypothesis) + delta))

    def numerical_diff(self, f, x):
        for i in range(len(x)):
            args = x[i]
            x[i] = args + self.h
            f1 = f()
            x[i] = args - self.h
            f2 = f()
            x[i] = args
            x[i] -= self.learning_rate * (f1 - f2) / (2 * self.h)


x = np.array([[2, 4], [4, 11], [6, 6], [8, 5], [10, 7], [12, 16], [14, 8], [16, 3], [18, 7]])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
nn = nn(x.T, y)
for i in range(800000):
    nn()
    if i % 10000 == 0:
        print(nn.w, nn.b)

print(nn.sigmoid(np.matmul(nn.w, nn.x) + nn.b))
