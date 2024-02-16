#define THREADS_PER_BLOCK 1024
#define DELLEXPORT extern "C"
const double e = 2.7182818284590452353602874713527;


template <typename T>
__global__ void tanh_impl_Kernel(T d_in, T d_out, int size){
	
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if(idx < size){
		// printf("%f\n", d_in[idx]);
		double sinh = (1 - pow(e, (-2 * d_in[idx]))) / (2 * pow(e, -d_in[idx]));
		double cosh = (1 + pow(e, (-2 * d_in[idx]))) / (2 * pow(e, -d_in[idx]));
		d_out[idx] = sinh / cosh;
		// d_out[idx] = ((1 - pow(e, (-2 * d_in[idx]))) / (2 * pow(e, -d_in[idx]))) / ((1 + pow(e, (-2 * d_in[idx]))) / (2 * pow(e, -d_in[idx])));
	}
}


DELLEXPORT void tanh_impl(double *, double *, int);