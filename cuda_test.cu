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

double* tanh_impl(double* x, int cnt){
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

int main()
{
	
	
	int cnt = 100000;

	double * x = new double[cnt];

	for(int i=0; i<cnt; ++i){
		x[i] = 1.;
	}
	printf("a %f CPU.\n", x[0]);
	double* a = tanh_impl(x, cnt);

	printf("a %f CPU.\n", a[0]);

	return 0;
}
