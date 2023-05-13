# define DELLEXPORT extern "C"

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#define THREADS_PER_BLOCK 1024
// nvcc -Xcompiler -fPIC -shared -g cuda_test.cu -o cuda_test.so
using namespace std ;
const double e = 2.7182818284590452353602874713527;



__global__ void tanh_impl_Kernel(double * d_in, double * d_out, int size){
	
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if(idx < size){
		// printf("%f\n", d_in[idx]);
		double sinh = (1 - pow(e, (-2 * d_in[idx]))) / (2 * pow(e, -d_in[idx]));
		double cosh = (1 + pow(e, (-2 * d_in[idx]))) / (2 * pow(e, -d_in[idx]));
		d_out[idx] = sinh / cosh;
		// d_out[idx] = ((1 - pow(e, (-2 * d_in[idx]))) / (2 * pow(e, -d_in[idx]))) / ((1 + pow(e, (-2 * d_in[idx]))) / (2 * pow(e, -d_in[idx])));
	}
}


DELLEXPORT void tanh_impl(double * h_in, double * h_out, int arr_size){
	
	const long long int ARRAY_BYTES = arr_size * sizeof(double);
	double *d_in, *d_out;

	cudaMalloc(&d_in, ARRAY_BYTES);
	cudaMalloc(&d_out, ARRAY_BYTES);

 	cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

	int block = ceil((float)arr_size/THREADS_PER_BLOCK);
	tanh_impl_Kernel<<< block, THREADS_PER_BLOCK >>>(d_in, d_out, arr_size);
	cudaDeviceSynchronize();
	cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

	cudaFree(d_in);
	cudaFree(d_out);
}


// int main()
// {
	
	
// 	int cnt = 100000;

// 	double * x = new double[cnt];

// 	for(int i=0; i<cnt; ++i){
// 		x[i] = 1.;
// 	}
// 	printf("a %f CPU.\n", x[0]);
// 	double* a = tanh_impl(x, cnt);

// 	printf("a %f CPU.\n", a[1]);

// 	return 0;
// }