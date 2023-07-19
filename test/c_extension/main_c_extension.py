import sys, os
a = os.system(f"make -C {os.path.dirname(__file__)}")
if a == 512: raise
from random import random
from time import perf_counter
from extension_test import _CPython_tanh_impl, _CPython_tanh_impl_point
from numpy_extension import _numpy_extension
import ctypes
import sys, os
from array import array
import numpy as np

lib_ctypes = ctypes.cdll.LoadLibrary(os.path.dirname(__file__) + '/ctypes_test.so')
lib_cuda = ctypes.cdll.LoadLibrary(os.path.dirname(__file__) + '/cuda_test.so')

COUNT = int(100000000/2) # Change this value depending on the speed of your computer
data = [(random() - 0.5) * 3 for _ in range(COUNT)]
data_numpy = np.array(data, dtype=np.float64)

def a(a):
    print(a.dtype)
    _numpy_extension(a)
    
a(data_numpy)
a(data_numpy.astype(np.float32))
a(data_numpy.astype(np.float16))

e = 2.7182818284590452353602874713527

def sinh(x):
    return (1 - (e ** (-2 * x))) / (2 * (e ** -x))

def cosh(x):
    return (1 + (e ** (-2 * x))) / (2 * (e ** -x))

def tanh(x):
    tanh_x = sinh(x) / cosh(x)
    return tanh_x

def python_tanh_impl():
    return [tanh(x) for x in data]


_ctypes_tanh_impl = lib_ctypes.ctypes_tanh_impl
_ctypes_tanh_impl.restype = ctypes.c_double
def ctypes_tanh_impl():
    return [_ctypes_tanh_impl(ctypes.c_double(x)) for x in data]


_ctypes_tanh_impl_point = lib_ctypes.ctypes_tanh_impl_point
_ctypes_tanh_impl_point.restype = ctypes.POINTER(ctypes.c_double * COUNT)
def ctypes_tanh_impl_point():
    temp = array('d', data)
    temp = (ctypes.c_double * len(data)).from_buffer(temp)
    result = _ctypes_tanh_impl_point(temp, len(data))
    return result.contents

def ctypes_tanh_impl_point_numpy():
    temp = data_numpy.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    result = _ctypes_tanh_impl_point(temp, len(data))
    result = np.ctypeslib.as_array(result.contents, shape=[COUNT])
    return result



_cuda_tanh_impl = lib_cuda.tanh_impl
_cuda_tanh_impl.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_size_t] 
def cuda_tanh_impl():
    temp = data_numpy.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    result = np.zeros_like(data)
    result_p = result.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    _cuda_tanh_impl(temp, result_p, len(data))
    return result

def CPython_tanh_impl():
    return [_CPython_tanh_impl(x) for x in data]

def CPython_tanh_impl_point():
    return _CPython_tanh_impl_point(data)


func = [cuda_tanh_impl, CPython_tanh_impl, CPython_tanh_impl_point, ctypes_tanh_impl_point, ctypes_tanh_impl_point_numpy, ctypes_tanh_impl, python_tanh_impl]

def test(fn):
    start = perf_counter()
    result = fn()
    duration = perf_counter() - start
    print(result[:5])
    print(f'{fn.__name__} took {duration:.3f} seconds\n')

if __name__ == "__main__":
    for i in func:
        test(i)