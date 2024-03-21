import lijnn
import numpy as np
arr = np.arange(36)

# 배열의 형태를 (3,3,3)으로 변경
arr = arr.reshape((3,3,4))

print(arr)
H, W = arr.shape[1], arr.shape[2]

argmin = np.argmin([H, W])

a = lijnn.datasets.VOCclassfication()[0]
print(a)
print()
