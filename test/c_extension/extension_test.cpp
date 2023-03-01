#include <cmath>
#include "python3.10/Python.h"

const double e = 2.7182818284590452353602874713527;

double sinh_impl(double x) {
    return (1 - pow(e, (-2 * x))) / (2 * pow(e, -x));
}

double cosh_impl(double x) {
    return (1 + pow(e, (-2 * x))) / (2 * pow(e, -x));
}

PyObject* CPython_tanh_impl(PyObject*, PyObject* o) {
    double x = PyFloat_AsDouble(o);
    double tanh_x = sinh_impl(x) / cosh_impl(x);
    return PyFloat_FromDouble(tanh_x);
}

PyObject* CPython_tanh_impl_point(PyObject*, PyObject* o) {
    Py_ssize_t tot_len = PyList_Size(o);
    PyObject* ret = PyList_New(tot_len);

    for (int i = 0; i < tot_len; i++) {
        double x = PyFloat_AsDouble(PyList_GetItem(o, i));
        double tanh_x = sinh_impl(x) / cosh_impl(x);
        PyList_SetItem(ret, i, PyFloat_FromDouble(tanh_x));
    }

    Py_INCREF(ret);

    return ret;
}

static PyMethodDef superfastcode_methods[] = {
    // The first property is the name exposed to Python, fast_tanh
    // The second is the C++ function with the implementation
    // METH_O means it takes a single PyObject argument
    { "CPython_tanh_impl", (PyCFunction)CPython_tanh_impl, METH_O, "asdfasdftestete" },
    { "CPython_tanh_impl_point", (PyCFunction)CPython_tanh_impl_point, METH_O, "asdfasdftestete" },

    // Terminate the array with an object containing nulls.
    { nullptr, nullptr, 0, nullptr }
};

static PyModuleDef superfastcode_module = {
    PyModuleDef_HEAD_INIT,
    "extension_test",                        // Module name to use with Python import statements
    "Provides some functions, but faster",  // Module description
    0,
    superfastcode_methods                   // Structure that defines the methods of the module
};

PyMODINIT_FUNC PyInit_extension_test() {
    return PyModule_Create(&superfastcode_module);
}