import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D

x = np.array([-50, -40, -30, -20, -10, -5, 0, 5, 10, 20, 30, 40, 50])
t = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

if x.ndim == 1: x = x[np.newaxis].T
if t.ndim == 1: t = t[np.newaxis].T


def rosenbrock(x0, x1):
    y = 100 * (x1 - x0 ** 2) ** 2 + (x0 - 1) ** 2
    return y


w = np.arange(-10000, 10000, 0.1).reshape(1,-1)
print(w,'\n',x)
b = np.arange(1, 10, 0.5).reshape(1,-1)
x1 =  np.sum((np.dot(x, w) - t) ** 2, axis=0) / 13

print(x1)



plt.plot(w.flat,x1)
plt.show()