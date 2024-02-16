#include <cmath>

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

extern "C" double ctypes_tanh_impl(double x){
    return tanh_impl(x);
}

extern "C" double* ctypes_tanh_impl_point(double* x, int cnt){
    double *result = new double[cnt];
    for (int i=0; i< cnt; i++){
        result[i] = std::tanh(x[i]);
    }
    return result;
}
