import numpy as np


def numerical_diff(dxindex, f, w):
    h = 1e-4  # 0.0001
    x1, x2 = list(w), list(w)
    x1[dxindex] += h
    x2[dxindex] -= h
    return (f(x1) - f(x2)) / (2 * h)

class linear:
    def __init__(self, x, y):
        self.h = 1e-4  # 0.0001
        self.learning_rate = 0.01
        self.x = x
        self.y = y
        if len(self.x.shape) == 1:
            self.x = self.x[np.newaxis]
        self.lensample = len(self.x[0])
        self.w = np.zeros(len(self.x))
        self.b = np.zeros(1)

    def __call__(self):
        cost = self.MSE(np.matmul(self.w, self.x) + self.b)
        for i in range(len(self.w)):
            x1, x2 = list(self.w), list(self.w)
            x1[i] += self.h
            x2[i] -= self.h
            f1 = self.MSE(np.matmul(x1, self.x) + self.b)
            f2 = self.MSE(np.matmul(x2, self.x) + self.b)
            self.w[i] -= self.learning_rate * (f1 - f2) / (2 * self.h)

        f1 = self.MSE(np.matmul(self.w, self.x) + self.b + self.h)
        f2 = self.MSE(np.matmul(self.w, self.x) + self.b - self.h)
        self.b -= self.learning_rate * (f1 - f2) / (2 * self.h)

        print(cost, self.b, self.w)


    def MSE(self, hypothesis):
        cost = np.sum((hypothesis - self.y) ** 2) / self.lensample
        return cost

x = np.array([1., 2., 3., 4., 5., 6.])
y = np.array([9., 16., 23., 30., 37., 44.])
linear = linear(x, y)
for i in range(100):
    linear()

print(linear.w[0]*3+linear.b)
