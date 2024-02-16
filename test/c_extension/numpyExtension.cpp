#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <cmath>
#include <Python.h>
#include "numpy/arrayobject.h"
#include <iostream>
#include "half.hpp"
#include <string>

using namespace std;
using half_float::half;

const double e = 2.7182818284590452353602874713527;

template <typename T>
class Variable {
public:
    typedef T value_type;
};

double sinh_impl(double x) {
    return (1 - pow(e, (-2 * x))) / (2 * pow(e, -x));
}

double cosh_impl(double x) {
    return (1 + pow(e, (-2 * x))) / (2 * pow(e, -x));
}

double tanh_impl(double x) {
    return sinh_impl(x) / cosh_impl(x);
}

template<typename T>
PyObject* main_f(PyArrayObject *arr){
    T *data;
    PyArray_AsCArray((PyObject **)&arr, &data, PyArray_DIMS(arr), PyArray_NDIM(arr), PyArray_DescrFromType(PyArray_TYPE(arr)));
    PyObject *result = PyArray_SimpleNew(PyArray_NDIM(arr), PyArray_DIMS(arr), PyArray_TYPE(arr));
    npy_intp size = PyArray_SIZE(arr);

    // 과학적 표기법을 소수점 표기법으로 표시
    // cout << fixed;

    // switch (PyArray_TYPE(arr))
    // {
    // case NPY_FLOAT16:
    //     cout.precision(3);
    //     break;
    // case NPY_FLOAT32:
    //     cout.precision(7);
    //     break;
    // case NPY_FLOAT64:
    //     cout.precision(16);
    //     break;
    // }
    // cout << ((T *)data)[0] << endl;

    T *result_data = (T *)PyArray_DATA((PyArrayObject *)result);
    for (npy_intp i = 0; i < size; ++i) {
        // result_data[i] = tanh_impl(data[i]);
        result_data[i] = std::tanh(data[i]);
    }
    
    return result;
}

PyObject* _numpy_extension(PyObject*, PyObject* args) {
    PyArrayObject *arr;
    PyArg_ParseTuple(args, "O", &arr);
    
    if (PyErr_Occurred()){
        return NULL;
    }

    int type = PyArray_TYPE(arr);
    PyObject *result;
    switch (type)
    {
    case NPY_FLOAT16:
        result = main_f<half>(arr);
        break;
    case NPY_FLOAT32:
        result = main_f<float>(arr);
        break;
    case NPY_FLOAT64:
        result = main_f<double>(arr);
        break;
    }
    return result;
}

static PyMethodDef methods[] = {
    // The first property is the name exposed to Python, fast_tanh
    // The second is the C++ function with the implementation
    // METH_O means it takes a single PyObject argument
    { "_numpy_extension", _numpy_extension, METH_VARARGS, "asdfasdftestete" },

    // Terminate the array with an object containing nulls.
    { nullptr, nullptr, 0, nullptr }
};

static PyModuleDef numpy_extension_module = {
    PyModuleDef_HEAD_INIT,
    "extension_test",                        // Module name to use with Python import statements
    "Provides some functions, but faster",  // Module description
    0,
    methods                   // Structure that defines the methods of the module
};

PyMODINIT_FUNC PyInit_numpyExtension() {
    import_array()
    return PyModule_Create(&numpy_extension_module);
}