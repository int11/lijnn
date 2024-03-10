import sys, os
makefile = os.system(f"make -C {os.path.dirname(__file__)}")
if makefile == 512: raise
from random import random, seed
from time import perf_counter
import cpythonExtension 
from numpyExtension import _numpy_extension
import ctypes
import sys, os
from array import array
import numpy as np

lib_ctypes = ctypes.cdll.LoadLibrary(os.path.dirname(__file__) + '/ctypes.so')
lib_cuda = ctypes.cdll.LoadLibrary(os.path.dirname(__file__) + '/cuda.so')


seed(1)
COUNT = int(50000000) # Change this value depending on the speed of your computer
data = [(random() - 0.5) * 3 for _ in range(COUNT)]
data_numpy = np.array(data, dtype=np.float64)

e = 2.7182818284590452353602874713527

def test_multitype(fun):
    a = [np.float64, np.float32, np.float16]
    for i in a:
        print(data_numpy.astype(i)[0])
        print(fun(data_numpy.astype(i))[0])

test_multitype(_numpy_extension)

def sinh(x):
    return (1 - (e ** (-2 * x))) / (2 * (e ** -x))

def cosh(x):
    return (1 + (e ** (-2 * x))) / (2 * (e ** -x))

def tanh(x):
    tanh_x = sinh(x) / cosh(x)
    return tanh_x

def python():
    return [tanh(x) for x in data]

def numpy():
    return np.tanh(data_numpy)

def numpyExtension():
    return _numpy_extension(data_numpy)


_ctypes_tanh_impl = lib_ctypes.tanh_impl
_ctypes_tanh_impl.restype = ctypes.c_double
def ctypesF(data):
    return [_ctypes_tanh_impl(ctypes.c_double(x)) for x in data]


_ctypes_tanh_impl_point = lib_ctypes.tanh_impl_point
_ctypes_tanh_impl_point.restype = ctypes.POINTER(ctypes.c_double * COUNT)
def ctypes_point0(data):
    ctypes_point_temp = array('d', data)
    ctypes_point_temp = (ctypes.c_double * len(data)).from_buffer(ctypes_point_temp)

    result = _ctypes_tanh_impl_point(ctypes_point_temp, len(data))
    return result.contents

def ctypes_point1(data):
    ctypes_point_numpy_temp = data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    
    result = _ctypes_tanh_impl_point(ctypes_point_numpy_temp, len(data))
    result = np.ctypeslib.as_array(result.contents, shape=[COUNT])
    return result

_cuda_tanh_impl = lib_cuda.tanh_impl
_cuda_tanh_impl.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_size_t] 
def cuda(data):
    cuda_cuda = data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    cuda_result = np.zeros_like(data)
    cuda_result_p0 = cuda_result.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
 
    _cuda_tanh_impl(cuda_cuda, cuda_result_p0, len(data))
    return cuda_result

def cpythonExtensionTanh(data):
    return [cpythonExtensionTanh.tanh_impl(x) for x in data]

def cpythonExtensionPointTanh0(data):
    output = []
    return cpythonExtension.tanh_impl_point0(data, output)

def cpythonExtensionPointTanh1(data):
    return cpythonExtension.tanh_impl_point1(data)

func = [cpythonExtensionPointTanh0, cpythonExtensionPointTanh1, cuda, numpy, numpyExtension, ctypes_point0, ctypes_point1]
functoslow = [cpythonExtensionTanh, ctypesF, python, ]

def test(fn):
    start = perf_counter()
    result = fn()
    duration = perf_counter() - start
        
    print(result[:5])
    print(f'{fn.__name__} took {duration:.3f} seconds\n')

if __name__ == "__main__":
    for i in func:
        test(i)