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

template <typename T>
T sinh_impl(T x) {
    return (1 - pow(e, (-2 * x))) / (2 * pow(e, -x));
}

template <typename T>
T cosh_impl(T x) {
    return (1 + pow(e, (-2 * x))) / (2 * pow(e, -x));
}

template <typename T>
void tanh_impl(void *data, npy_intp size, T *result){
    for (int i=0; i < size; ++i){
        T x =  ((T *)data)[i];
        result[i] = sinh_impl(x) / cosh_impl(x);
    }
}

template<typename T>
void main_f(PyArrayObject *arr){
    void *data;
    PyArray_AsCArray((PyObject **)&arr, &data, PyArray_DIMS(arr), PyArray_NDIM(arr), PyArray_DescrFromType(PyArray_TYPE(arr)));
    PyObject *result = PyArray_SimpleNew(PyArray_NDIM(arr), PyArray_DIMS(arr), PyArray_TYPE(arr));
    npy_intp size = PyArray_SIZE(arr);
    cout << fixed;
    
    switch (PyArray_TYPE(arr))
    {
    case NPY_FLOAT16:
        cout.precision(16);
        break;
    case NPY_FLOAT32:
        cout.precision(7);
        break;
    case NPY_FLOAT64:
        cout.precision(3);
        break;
    }
    cout << "c : " << ((T *)data)[0] << endl;
}

PyObject* _numpy_extension(PyObject*, PyObject* args) {
    PyArrayObject *arr;
    PyArg_ParseTuple(args, "O", &arr);
    
    if (PyErr_Occurred()){
        return NULL;
    }

    int type = PyArray_TYPE(arr);
    switch (type)
    {
    case NPY_FLOAT16:
        main_f<half>(arr);
        break;
    case NPY_FLOAT32:
        main_f<float>(arr);
        break;
    case NPY_FLOAT64:
        main_f<double>(arr);
        break;
    }
    return args;
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

PyMODINIT_FUNC PyInit_numpy_extension() {
    import_array()
    return PyModule_Create(&numpy_extension_module);
}