import numpy as np


def numerical_diff(dxindex, f, w):
    h = 1e-4  # 0.0001
    x1, x2 = list(w), list(w)
    x1[dxindex] += h
    x2[dxindex] -= h
    return (f(x1) - f(x2)) / (2 * h)


def linear(w):
    return w[0] * datax + w[1]


def MSEmaker(f):
    def MSE(w):
        return np.sum((datay - f(w)) ** 2) / lenx

    return MSE


datax = np.array([1., 2., 3., 4., 5., 6.])
datay = np.array([9., 16., 23., 30., 37., 44.])
lenx = len(datax)

w = [0.,0.]
learning_rate = 0.01

MSE = MSEmaker(linear)


for i in range(500000):
    cost = MSE(w)
    w[0] -= learning_rate * numerical_diff(0, MSE, w)
    w[1] -= learning_rate * numerical_diff(1, MSE, w)
    if i % 100 == 0:
        print('Epoch ({:10d}/{:10d}) cost: {:10f}, W: {:10f}, b:{:10f}'.format(i, 500, cost, w[0], w[1]))