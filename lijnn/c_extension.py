import ctypes
import sys, os

libc = ctypes.cdll.LoadLibrary(os.path.dirname(__file__) + '/ctypes_test.so')

ctypes_tanh_impl_point = libc.ctypes_tanh_impl_point
ctypes_tanh_impl_point.restype = ctypes.POINTER(ctypes.c_double)

def ctypes_pointer_numpy(data):
    temp = np.ctypeslib.as_ctypes(np.array(data))
    result = ctypes_tanh_impl_point(temp, len(data))
    result = np.ctypeslib.as_array(result, shape=[COUNT])
    return result
