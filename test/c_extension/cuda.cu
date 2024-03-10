#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include "cuda.cuh"
using namespace std ;


void tanh_impl(double * h_in, double * h_out, int arr_size){
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