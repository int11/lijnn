from lijnn import *
import numpy as np
import cupy as cp
from lijnn.functions import *


x = Variable(np.array([3]))
y = add(x, x)

y.backward(retain_grad=True)

print(y.grad.data == 1 and x.grad.data == 2)
