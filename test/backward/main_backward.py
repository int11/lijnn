import sys, os
sys.path.append(os.getcwd())

from lijnn import *
import numpy as np
from lijnn.functions import *


x = Variable(np.array([3]))
y = add(x, x)

y.backward(retain_grad=True)

print(y.grad.data == 1)
print(x.grad.data == 2)