#include <cmath>

const double e = 2.7182818284590452353602874713527;

extern "C" double sinh_impl(double x) {
    return (1 - pow(e, (-2 * x))) / (2 * pow(e, -x));
}

extern "C" double cosh_impl(double x) {
    return (1 + pow(e, (-2 * x))) / (2 * pow(e, -x));
}

extern "C" double tanh_impl(double x){
    return sinh_impl(x) / cosh_impl(x);;
}

extern "C" double* tanh_impl_point(double* x, int cnt){
    double *result = new double[cnt];
    for (int i=0; i< cnt; i++){
        result[i] = sinh_impl(x[i]) / cosh_impl(x[i]);
    }
    return result;
}
