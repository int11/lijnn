#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <cmath>
#include <Python.h>
#include "numpy/arrayobject.h"

const double e = 2.7182818284590452353602874713527;

double sinh_impl(double x) {
    return (1 - pow(e, (-2 * x))) / (2 * pow(e, -x));
}

double cosh_impl(double x) {
    return (1 + pow(e, (-2 * x))) / (2 * pow(e, -x));
}

PyObject* _numpy_extension(PyObject*, PyObject* o) {
    PyArrayObject *arr = (PyArrayObject *)o;
    auto *a = PyArray_DATA(arr);
}

static PyMethodDef methods[] = {
    // The first property is the name exposed to Python, fast_tanh
    // The second is the C++ function with the implementation
    // METH_O means it takes a single PyObject argument
    { "_numpy_extension", _numpy_extension, METH_O, "asdfasdftestete" },

    // Terminate the array with an object containing nulls.
    { nullptr, nullptr, 0, nullptr }
};

static PyModuleDef numpy_extensiontest_module = {
    PyModuleDef_HEAD_INIT,
    "extension_test",                        // Module name to use with Python import statements
    "Provides some functions, but faster",  // Module description
    0,
    methods                   // Structure that defines the methods of the module
};

PyMODINIT_FUNC PyInit_numpy_extensiontest() {
    return PyModule_Create(&numpy_extensiontest_module);
}
