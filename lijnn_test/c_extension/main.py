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

class speedTest:
    seed(1)
    COUNT = int(50000000 *2) # Change this value depending on the speed of your computer

    def __init__(self, data_type):
        self.data_type = data_type

    def __call__(self, func):
        def wrapper():
            if self.data_type == "list":
                data = [(random() - 0.5) * 3 for _ in range(self.COUNT)]
            elif self.data_type == "numpy":
                data = [(random() - 0.5) * 3 for _ in range(self.COUNT)]
                data = np.array(data, dtype=np.float64)
            elif self.data_type == "ctypes":
                data = [(random() - 0.5) * 3 for _ in range(self.COUNT)]
                data = np.array(data, dtype=np.float64)

                '''data.ctypes.data_as(ctypes.POINTER(type)): 이 메서드는 numpy 배열의 데이터를 특정 ctypes 포인터 타입으로 해석합니다. 
                type은 원하는 ctypes 데이터 타입이며, 이는 numpy 배열의 데이터 타입과 일치해야 합니다.
                예를 들어, float32 numpy 배열의 경우 ctypes.c_float를 사용해야 합니다.
                numpy 배열의 데이터를 가리키는 ctypes 포인터를 반환'''
                # data = data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                

                '''np.ctypeslib.as_ctypes(data): 이 함수는 numpy 배열을 입력으로 받아 
                해당 배열에 대응하는 새로운 ctypes 배열을 생성합니다.
                반환되는 ctypes 배열은 numpy 배열과 동일한 메모리를 공유하므로, 한 배열에서의 변경이 다른 배열에도 반영됩니다.
                ctypes 배열을 반환
                아래 주석 함수는 list 를 받아 똑같은 ctypes 배열을 반환'''
                # ctypes_point_temp = array('d', data)
                # ctypes_point_temp = (ctypes.c_double * len(data)).from_buffer(ctypes_point_temp)
                data = np.ctypeslib.as_ctypes(data)

            start = perf_counter()
            result = func(data)
            duration = perf_counter() - start
            print(result[:5])
            print(f'{func.__name__} took {duration:.3f} seconds\n')
            return result, duration
        
        return wrapper
    

lib_ctypes = ctypes.cdll.LoadLibrary(os.path.dirname(__file__) + '/ctypes.so')
lib_cuda = ctypes.cdll.LoadLibrary(os.path.dirname(__file__) + '/cuda.so')

_ctypes_tanh_impl = lib_ctypes.tanh_impl
_ctypes_tanh_impl.restype = ctypes.c_double

_ctypes_tanh_impl_point = lib_ctypes.tanh_impl_point
_ctypes_tanh_impl_point.restype = ctypes.POINTER(ctypes.c_double * speedTest.COUNT)

_cuda_tanh_impl = lib_cuda.tanh_impl
_cuda_tanh_impl.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]

e = 2.7182818284590452353602874713527

def test_multitype(fun):
    a = [np.float64, np.float32, np.float16]
    for i in a:
        print(data_numpy.astype(i)[0])
        print(fun(data_numpy.astype(i))[0])



def sinh(x):
    return (1 - (e ** (-2 * x))) / (2 * (e ** -x))

def cosh(x):
    return (1 + (e ** (-2 * x))) / (2 * (e ** -x))

def tanh(x):
    tanh_x = sinh(x) / cosh(x)
    return tanh_x

@speedTest("list")
def python(data):
    return [tanh(x) for x in data]

@speedTest("numpy")
def numpy(data):
    return np.tanh(data)

@speedTest("numpy")
def numpyExtension(data):
    return _numpy_extension(data)

@speedTest("ctypes")
def ctypesF(data):
    return [_ctypes_tanh_impl(ctypes.c_double(x)) for x in data]

@speedTest("ctypes")
def ctypes_point(data):
    result = _ctypes_tanh_impl_point(data, len(data))
    result = np.ctypeslib.as_array(result.contents)
    return result

@speedTest("ctypes")
def cuda(data):
    result = np.zeros_like(data.__keep)
    result_ctypes = np.ctypeslib.as_ctypes(result)
    _cuda_tanh_impl(data, result_ctypes, len(data))
    return result

@speedTest("list")
def cpythonExtensionTanh(data):
    return [cpythonExtensionTanh.tanh_impl(x) for x in data]

@speedTest("list")
def cPythonExtensionPointTanh0(data):
    output = []
    return cpythonExtension.tanh_impl_point0(data, output)

@speedTest("list")
def cPythonExtensionPointTanh1(data):
    return cpythonExtension.tanh_impl_point1(data)

func = [cuda, numpy, numpyExtension, ctypes_point, cPythonExtensionPointTanh0, cPythonExtensionPointTanh1]
functoslow = [cpythonExtensionTanh, ctypesF, python]

if __name__ == "__main__":
    for i in func:
        i()