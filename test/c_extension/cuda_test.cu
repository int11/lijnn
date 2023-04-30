#include <iostream>
#include <cuda.h>

using namespace std ;

# define DELLEXPORT extern "C"


#include <iostream>
#include <cuda.h>
#include <cmath>
#define THREADS_PER_BLOCK 512
// nvcc -Xcompiler -fPIC -shared -g cuda_test.cu -o cuda_test.so
using namespace std ;
const double e = 2.7182818284590452353602874713527;


__global__ void tanh_impl_Kernel(double *d_x, int cnt, double *d_result){
	int idx =threadIdx.x + blockIdx.x * blockDim.x;
    double sinh = (1 - pow(e, (-2 * d_x[idx]))) / (2 * pow(e, -d_x[idx]));
    double cosh = (1 + pow(e, (-2 * d_x[idx]))) / (2 * pow(e, -d_x[idx]));

	d_result[idx] = sinh / cosh;
}

extern "C" double* tanh_impl(double* x, int cnt){
    double *result =(double *)malloc(cnt);
    double *d_x, *d_result;

    cudaMalloc(&d_x, cnt);
    cudaMalloc(&d_result, cnt);

    cudaMemcpy(d_x, x, cnt, cudaMemcpyHostToDevice);
    


    tanh_impl_Kernel <<< cnt / THREADS_PER_BLOCK , THREADS_PER_BLOCK >>>(d_x, cnt, d_result); 


    cudaMemcpy(result, d_result, cnt, cudaMemcpyDeviceToHost); 

    cudaFree(d_x); 
    cudaFree(d_result); 
    
    return result;
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


__global__ void cudaSquareKernel(float * d_in, float * d_out){
	
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	printf("threadIdx.x : %d, blockIdx.x : %d, blockDim.x : %d \n", threadIdx.x, blockIdx.x, blockDim.x);
	d_out[idx] = d_in[idx] * d_in[idx] ;
}

DELLEXPORT float* cudaSquare(float * h_in, int arr_size){
	
	const long long int ARRAY_BYTES = arr_size * sizeof(float) ;
    float * h_out = (float *)malloc(arr_size);
	float *d_in, *d_out ;

	cudaMalloc(&d_in, ARRAY_BYTES) ;
	cudaMalloc(&d_out, ARRAY_BYTES) ;

	cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice) ;
	
	cudaSquareKernel<<< 1, arr_size >>>(d_in, d_out) ;

	cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost) ;

	cudaFree(d_in) ;
	cudaFree(d_out) ;
    return h_out;
}