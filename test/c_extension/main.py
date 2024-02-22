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


seed(0)
COUNT = int(1000000) # Change this value depending on the speed of your computer
data = [(random() - 0.5) * 3 for _ in range(COUNT)]
data_numpy = np.array(data, dtype=np.float64)

e = 2.7182818284590452353602874713527

output = [1.4]
a = cpythonExtension.tanh_impl_point(data, output)
print(id(output), id(cpythonExtension.my_function(output)))
print(id(data), id(a), a[0], )


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
def ctypesF():
    return [_ctypes_tanh_impl(ctypes.c_double(x)) for x in data]


_ctypes_tanh_impl_point = lib_ctypes.tanh_impl_point
_ctypes_tanh_impl_point.restype = ctypes.POINTER(ctypes.c_double * COUNT)
def ctypes_point():
    ctypes_point_temp = array('d', data)
    ctypes_point_temp = (ctypes.c_double * len(data)).from_buffer(ctypes_point_temp)

    start = perf_counter()
    result = _ctypes_tanh_impl_point(ctypes_point_temp, len(data))
    return result.contents, perf_counter() - start

def ctypes_point_numpy():
    ctypes_point_numpy_temp = data_numpy.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    
    start = perf_counter()
    result = _ctypes_tanh_impl_point(ctypes_point_numpy_temp, len(data))
    result = np.ctypeslib.as_array(result.contents, shape=[COUNT])
    return result, perf_counter() - start

_cuda_tanh_impl = lib_cuda.tanh_impl
_cuda_tanh_impl.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_size_t] 
def cuda():
    cuda_cuda = data_numpy.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    cuda_result = np.zeros_like(data)
    cuda_result_p0 = cuda_result.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    start = perf_counter()
    _cuda_tanh_impl(cuda_cuda, cuda_result_p0, len(data))
    return cuda_result, perf_counter() - start




def cpythonExtensionTanh():
    return [cpythonExtensionTanh.tanh_impl(x) for x in data]

def cpythonExtensionPointTanh():
    return cpythonExtension.tanh_impl_point(data)


func = [cuda, numpy, numpyExtension, cpythonExtensionPointTanh, ctypes_point, ctypes_point_numpy]
functoslow = [cpythonExtensionTanh, ctypesF, python]
def test(fn):
    start = perf_counter()
    result = fn()
    if type(result) is tuple:
        duration = result[1]
        result = result[0]
    else:
        duration = perf_counter() - start
        
    print(result[:5])
    print(f'{fn.__name__} took {duration:.3f} seconds\n')

if __name__ == "__main__":
    for i in func:
        test(i)