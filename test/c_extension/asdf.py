
import numpy as np
import ctypes
from ctypes import * 
import os
size = int(1024) 

def get_cuda_square():
	dll = ctypes.cdll.LoadLibrary(os.path.dirname(__file__) + '/cuda_test.so')
	func = dll.cudaSquare
	func.argtypes = [POINTER(c_float), c_size_t] 
	func.restype = ctypes.POINTER(ctypes.c_float * size)
	return func

__cuda_square = get_cuda_square()

def cuda_square(a, size):
	a_p = a.ctypes.data_as(POINTER(c_float))

	return __cuda_square(a_p, size)

if __name__ == '__main__':
	

	a = np.arange(1, size + 1).astype('float32')
	b = cuda_square(a, size)
	b = b.contents

	for i in range(size):
		print(b[i], end = "")
		print( '\t' if ((i % 4) != 3) else "\n", end = " ", flush = True)