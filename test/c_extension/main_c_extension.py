import sys, os

from random import random
from time import perf_counter
#g++ -shared -g extension_test.cpp -o extension_test.so
from extension_test import CPython_tanh_impl, CPython_tanh_impl_point
#g++ -shared -g ctypes_test.cpp -o ctypes_test.so
import ctypes
import sys, os
from array import array
import numpy as np

lib_ctypes = ctypes.cdll.LoadLibrary(os.path.dirname(__file__) + '/ctypes_test.so')

lib_cuda = ctypes.cdll.LoadLibrary(os.path.dirname(__file__) + '/cuda_test.so')

COUNT = 10000000  # Change this value depending on the speed of your computer
data = [(random() - 0.5) * 3 for _ in range(COUNT)]

e = 2.7182818284590452353602874713527

def sinh(x):
    return (1 - (e ** (-2 * x))) / (2 * (e ** -x))

def cosh(x):
    return (1 + (e ** (-2 * x))) / (2 * (e ** -x))

def tanh(x):
    tanh_x = sinh(x) / cosh(x)
    return tanh_x

def test(fn, name):
    start = perf_counter()
    result = fn(data)
    duration = perf_counter() - start
    print(f'{name} took {duration:.3f} seconds\n\n')

    for d in result:
        assert -1 <= d <= 1, " incorrect values"

ctypes_tanh_impl = lib_ctypes.ctypes_tanh_impl
ctypes_tanh_impl.restype = ctypes.c_double

ctypes_tanh_impl_point = lib_ctypes.ctypes_tanh_impl_point
ctypes_tanh_impl_point.restype = ctypes.POINTER(ctypes.c_double * COUNT)

cuda_tanh_impl = lib_cuda.tanh_impl
cuda_tanh_impl.restype = ctypes.POINTER(ctypes.c_double * COUNT)

def ctypes_pointer(data):
    temp = array('d', data)
    temp = (ctypes.c_double * len(data)).from_buffer(temp)
    result = ctypes_tanh_impl_point(temp, len(data))
    return result.contents

def ctypes_pointer_numpy(data):
    temp = np.ctypeslib.as_ctypes(np.array(data))
    result = ctypes_tanh_impl_point(temp, len(data))
    result = np.ctypeslib.as_array(result.contents, shape=[COUNT])
    return result

def cuda_tanh_impl1(data):
    temp = np.ctypeslib.as_ctypes(np.array(data))
    result = cuda_tanh_impl(temp, len(data))
    result = np.ctypeslib.as_array(result.contents, shape=[COUNT])
    return result

if __name__ == "__main__":
    print(ctypes_tanh_impl(ctypes.c_double(data[1])), ctypes_pointer(data)[1], ctypes_pointer_numpy(data)[1])
    print(CPython_tanh_impl(data[1]), CPython_tanh_impl_point(data)[1])
    print(tanh(data[1]))

    print(cuda_tanh_impl1(data)[1])

    print('Running benchmarks with COUNT = {}'.format(COUNT))


    test(lambda d: [ctypes_tanh_impl(ctypes.c_double(x)) for x in d], 'ctypes')
    test(lambda d: ctypes_pointer(data), 'ctypes pointer')
    test(lambda d: ctypes_pointer_numpy(data), 'ctypes pointer numpy')

    test(lambda d: [CPython_tanh_impl(x) for x in d], 'CPython C++ extension')
    test(lambda d: CPython_tanh_impl_point(data), 'CPython C++ extension pointer')
    
    test(lambda d: [tanh(x) for x in d], 'Python implementation')