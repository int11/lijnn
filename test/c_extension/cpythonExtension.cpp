#include <iostream>
#include <cmath>
#include <Python.h>
#include <vector>

using namespace std;

const double e = 2.7182818284590452353602874713527;

double sinh_impl(double x) {
    return (1 - pow(e, (-2 * x))) / (2 * pow(e, -x));
}

double cosh_impl(double x) {
    return (1 + pow(e, (-2 * x))) / (2 * pow(e, -x));
}

double tanh_impl(double x) {
    return sinh_impl(x) / cosh_impl(x);
}

PyObject* tanh_impl(PyObject*, PyObject* o) {
    double x = PyFloat_AsDouble(o);
    double tanh_x = tanh_impl(x);
    return PyFloat_FromDouble(tanh_x);
}

PyObject* tanh_impl_point0(PyObject*, PyObject* o) {
    PyObject *input, *output;
    if (!PyArg_ParseTuple(o, "OO", &input, &output)) {
        return NULL;
    }
    Py_ssize_t size = PyList_Size(input);
    
    for (int i = 0; i < size; i++) {
        double x = PyFloat_AsDouble(PyList_GetItem(input, i));
        double tanh_x = std::tanh(x);
        PyList_Append(output, PyFloat_FromDouble(tanh_x));
    }
    Py_INCREF(output);
    return output;
}

PyObject* tanh_impl_point1(PyObject*, PyObject* o) {
    
    PyObject *input;
    if (!PyArg_ParseTuple(o, "O", &input)) {
        return NULL;
    }
    Py_ssize_t size = PyList_Size(input);
    PyObject* output = PyList_New(0);
    Py_INCREF(output);
    for (int i = 0; i < size; i++) {
        double x = PyFloat_AsDouble(PyList_GetItem(input, i));
        double tanh_x = std::tanh(x);
         PyList_Append(output, PyFloat_FromDouble(tanh_x));
    }

    return output;
}
static PyMethodDef methods[] = {
    // The first property is the name exposed to Python, fast_tanh
    // The second is the C++ function with the implementation
    // METH_O means it takes a single PyObject argument
    { "tanh_impl", tanh_impl, METH_O, "asdfasdftestete" },
    { "tanh_impl_point0", tanh_impl_point0, METH_VARARGS, "asdfasdftestete" },
    { "tanh_impl_point1", tanh_impl_point1, METH_VARARGS, "asdfasdftestete" },
    // Terminate the array with an object containing nulls.
    { nullptr, nullptr, 0, nullptr }
};

static PyModuleDef extension_test_module = {
    PyModuleDef_HEAD_INIT,
    "extension_test",                        // Module name to use with Python import statements
    "Provides some functions, but faster",  // Module description
    0,
    methods                   // Structure that defines the methods of the module
};

PyMODINIT_FUNC PyInit_cpythonExtension() {
    return PyModule_Create(&extension_test_module);
}
