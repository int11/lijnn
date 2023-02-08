import sys, os
sys.path.append(os.getcwd())

from random import random
from time import perf_counter
from extension_test import fast_tanh
import ctypes
import sys, os
from array import array
libc = ctypes.cdll.LoadLibrary(os.path.dirname(__file__) + '/ctypes_test.so')
print(libc)
COUNT = 5000000  # Change this value depending on the speed of your computer
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

    for d in result[:5000000]:
        assert -1 <= d <= 1, " incorrect values"

if __name__ == "__main__":
    tanh_impl = libc.tanh_impl
    sinh_impl = libc.sinh_impl
    tanh_impl_point = libc.tanh_impl_point
    tanh_impl.restype = ctypes.c_double
    sinh_impl.restype = ctypes.c_double
    tanh_impl_point.restype = ctypes.POINTER(ctypes.c_double)

    print(data[1], ctypes.c_double(data[1]))
    print(sinh_impl(ctypes.c_double(data[1])), sinh(data[1]))
    print()
    print(tanh(data[1]),fast_tanh(data[1]))
    print(tanh_impl(ctypes.c_double(data[1])))

    

    print('Running benchmarks with COUNT = {}'.format(COUNT))

    test(lambda d: [tanh(x) for x in d], '[tanh(x) for x in d] (Python implementation)')
    test(lambda d: [fast_tanh(x) for x in d], '[fast_tanh(x) for x in d] (CPython C++ extension)')
    test(lambda d: [tanh_impl(ctypes.c_double(x)) for x in d], '[fast_tanh(x) for x in d] (ctypes)')

    def aa(data0):
        temp = array('d', data0)
        temp = (ctypes.c_double * len(data0)).from_buffer(temp)
        return tanh_impl_point(temp, len(data0))

    test(lambda d: aa(data), 'tanh_impl_point')

    