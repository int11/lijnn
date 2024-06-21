import lijnn
import numpy as np
from lijnn.utils import *
import lijnn.functions_conv as F


img = np.zeros((1,1,4,4))
np.add.at(img, [[0],[0],[0],[0]], 1)
print(img)
a = Variable(np.random.randn(1,1,4,4).astype(np.float32))

b = F.im2col(a, 2, to_matrix=False)
b.backward()
