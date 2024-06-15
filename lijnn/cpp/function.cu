#include <iostream>
#include <cuda.h>
#include <cudnn.h>
using namespace std;

void checkCUDNN(cudnnStatus_t status)
{
    if (status != CUDNN_STATUS_SUCCESS)
        cout << "[ERROR] CUDNN " << status << endl;
}

void checkCUDA(cudaError_t error)
{
    if (error != CUDA_SUCCESS)
        cout << "[ERROR] CUDA " << error << endl;
}

int main()
{
    const int batch_count = 1; //입력 데이터 갯수, 배치사이즈
    const int in_channel = 2; //입력 데이터의 채널 수
    const int in_height = 4; //입력 데이터의 세로 길이
    const int in_width = 4; //입력 데이터의 가로 길이
    const int filter_width = 3; //컨볼루션 필터(가중치)의 가로 길이
    const int filter_height = 3; //컨볼루션 필터(가중치)의 세로 길이
    const int filter_num = 1; //컨볼루션 필터(가중치) 갯수
    const int padding_w = 1; //컨볼루션 패딩. 필터의 가로 세로 길이가 3이고 패딩이 1,1 이면 SAME Convolution이 된다
    const int padding_h = 1;
    const int stride_horizontal = 1; //컨볼루션 스트라이드
    const int stride_vertical = 1;
    float inData[batch_count][in_channel][in_height][in_width]; //host 입력 데이터
    float outData[batch_count][filter_num][in_height][in_width]; //host 출력 데이터
    float *inData_d; //device 입력 데이터
    float *outData_d; //device 출력 데이터
    float *filterData_d; //device 컨볼루션 필터 데이터
    void *workSpace; //CUDNN이 작업 중에 사용할 버퍼 메모리

    //입력 데이터 셋팅
    for (int i = 0; i < in_channel; i++)
    {
        for (int y = 0; y < in_height; y++)
        {
            for (int x = 0; x < in_width; x++)
            {
                inData[0][i][y][x] = i * in_channel * in_height * in_width + y * in_width + x;
            }
        }
    }

    //필터(가중치) 셋팅
    float filterData[filter_num][in_channel][filter_height][filter_width] = {
        {
            {{0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 0.0f}},
            {{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 0.0f}}}};
    //GPU 메모리 할당
    checkCUDA(cudaMalloc((void **)&inData_d, sizeof(inData)));
    checkCUDA(cudaMalloc((void **)&outData_d, sizeof(outData)));;
    checkCUDA(cudaMalloc((void **)&filterData_d, sizeof(filterData)));

    //CPU 데이터를 GPU 메모리로 복사
    checkCUDA(cudaMemcpy(inData_d, inData, sizeof(inData), cudaMemcpyHostToDevice));
    checkCUDA(cudaMemcpy(filterData_d, filterData, sizeof(filterData), cudaMemcpyHostToDevice));
    //CUDNN 배열
    cudnnHandle_t cudnnHandle; // CUDNN을 사용하기 위한 핸들러
    cudnnTensorDescriptor_t inTensorDesc, outTensorDesc; //입력데이터와 출력데이터 구조체 선언
    cudnnFilterDescriptor_t filterDesc; //필터 구조체 선언
    cudnnConvolutionDescriptor_t convDesc; //컨볼루션 구조체 선언
    //할당
    checkCUDNN(cudnnCreate(&cudnnHandle));
    checkCUDNN(cudnnCreateTensorDescriptor(&inTensorDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&outTensorDesc));
    checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));

    //초기화
    //데이터의 자료형, 구조가 [Number][Channel][Height][Width] 형태임을 알려줌
    checkCUDNN(cudnnSetTensor4dDescriptor(inTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_count, in_channel, in_height, in_width));
    checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, filter_num, in_channel, filter_height, filter_width));
    //컨볼루션의 패딩, 스트라이드, 컨볼루션 모드 등을 셋팅

    //cuDNN9
    //checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc, padding_h, padding_w, stride_vertical, stride_horizontal, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    //cuDNN5
    checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc, padding_h, padding_w, stride_vertical, stride_horizontal, 1, 1, CUDNN_CROSS_CORRELATION));

    int out_n, out_c, out_h, out_w;
    //입력데이터를 위에서 셋팅한 대로 컨볼루션 했을때 출력 데이터의 구조 알아내기
    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc, inTensorDesc, filterDesc, &out_n, &out_c, &out_h, &out_w));
    printf("conv out shape (n x c x h x w) = (%d x %d x %d x %d)\n", out_n, out_c, out_h, out_w);
    //출력 데이터의 자료형, 구조를 셋팅
    checkCUDNN(cudnnSetTensor4dDescriptor(outTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, out_n, out_c, out_h, out_w));
    //입력과 필터, 컨볼루션 패딩, 스트라이드가 위와 같이 주어졌을때 가장 빠르게 계산할 수 있는 알고리즘이 무엇인지를 알아내기
    cudnnConvolutionFwdAlgo_t algo;
    checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnnHandle,
                                                   inTensorDesc,
                                                   filterDesc,
                                                   convDesc,
                                                   outTensorDesc,
                                                   CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                   0,
                                                   &algo));

    cout << "Fastest algorithm (enum cudnnConvolutionFwdAlgo_t) = " << algo << endl;
    //위에서 알아낸 가장 빠른 알고리즘을 사용할 경우 계산과정에서 필요한 버퍼 데이터의 크기를 알아내기
    size_t sizeInBytes = 0;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
                                                       inTensorDesc,
                                                       filterDesc,
                                                       convDesc,
                                                       outTensorDesc,
                                                       algo,
                                                       &sizeInBytes));

    cout << "sizeInBytes " << sizeInBytes << endl;
    //계산과정에서 버퍼 데이터가 필요하다면 메모리 할당
    if (sizeInBytes != 0)
        checkCUDA(cudaMalloc(&workSpace, sizeInBytes));
    float alpha = 1;
    float beta = 0;
    //컨볼루션 시작
    //alpha와 beta는 "output = alpha * Op(inputs) + beta * output" 에 사용됨
    //일반 컨볼루션은 output =   1   *  inputs * W
    //그래서          output =   1   * Op(inputs) +   0  * output 이 되도록 alpha와 beta를 1,0으로 셋팅함
    checkCUDNN(cudnnConvolutionForward(cudnnHandle,
                                       &alpha,
                                       inTensorDesc,
                                       inData_d,
                                       filterDesc,
                                       filterData_d,
                                       convDesc,
                                       algo,
                                       workSpace,
                                       sizeInBytes,
                                       &beta,
                                       outTensorDesc,
                                       outData_d));

    //컨볼루션 결과를 host로 복사
    checkCUDA(cudaMemcpy(outData, outData_d, sizeof(outData), cudaMemcpyDeviceToHost));

    cout << "in" << endl;
    for (int i = 0; i < in_channel; i++)
    {
        for (int y = 0; y < in_height; y++)
        {
            for (int x = 0; x < in_width; x++)
            {
                printf("%.0f ", inData[0][i][y][x]);
            }
            cout << endl;
        }
        cout << endl;
    }

    cout << "weights" << endl;
    for (int n = 0; n < filter_num; n++)
    {
        for (int i = 0; i < in_channel; i++)
        {
            for (int y = 0; y < filter_height; y++)
            {
                for (int x = 0; x < filter_width; x++)
                {
                    printf("%.1f ", filterData[n][i][y][x]);
                }
                cout << endl;
            }
            cout << endl;
        }
    }

    cout << "out" << endl;
    for (int i = 0; i < filter_num; i++)
    {
        for (int y = 0; y < in_height; y++)
        {
            for (int x = 0; x < in_width; x++)
            {
                printf("%.1f ", outData[0][i][y][x]);
            }
            cout << endl;
        }
        cout << endl;
    }
    //메모리 해제
    checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
    checkCUDNN(cudnnDestroyFilterDescriptor(filterDesc));
    checkCUDNN(cudnnDestroyTensorDescriptor(inTensorDesc));
    checkCUDNN(cudnnDestroyTensorDescriptor(outTensorDesc));
    checkCUDNN(cudnnDestroy(cudnnHandle));
    checkCUDA(cudaFree(inData_d));
    checkCUDA(cudaFree(outData_d));;
    checkCUDA(cudaFree(filterData_d));
    return 0;
}