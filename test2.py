import numpy as np

x = np.array([2, 5, 10, 20, 30, 40, 50])
w = np.array([3, 4, 10, 20, 30, 40, 50]).reshape(1,-1)
if x.ndim == 1: x = x[np.newaxis].T

print(x,'\n',w)
print(np.dot(x, w))
print(x*w)