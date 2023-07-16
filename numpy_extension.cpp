#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <cmath>
#include <Python.h>
#include "numpy/arrayobject.h"
#include <iostream>
using namespace std;

const double e = 2.7182818284590452353602874713527;

double sinh_impl(double x) {
    return (1 - pow(e, (-2 * x))) / (2 * pow(e, -x));
}

double cosh_impl(double x) {
    return (1 + pow(e, (-2 * x))) / (2 * pow(e, -x));
}

PyObject* _numpy_extension(PyObject*, PyObject* args) {
    PyArrayObject *arr;
    PyArg_ParseTuple(args, "O", &arr);
    
    
    if (PyErr_Occurred()){
        return NULL;
    }

    double *data = (double *)PyArray_DATA(arr);
    npy_intp size = PyArray_SIZE(arr);
    PyArray_Descr * dtype = PyArray_DTYPE(arr);

    for (int i = 0; i < size; ++i){
        double tanh_x = sinh_impl(data[i]) / cosh_impl(data[i]);
        printf("%f", tanh_x);
    }
    return NULL;
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
