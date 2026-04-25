#include <stdio.h>
#include <stdlib.h>

// 检查 CUDA 错误的宏（相当于之前用的 HANDLE_ERROR）
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

__global__ void kernel1(int M,int N,int K,float* A,float* B,float* C){
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = threadIdx.y + blockIdx.y * blockDim.y;
  
    if(row < M && col < N){
        float acc = 0.0f;
        for(int k = 0;k < K; ++k){
            acc += A[k * M + row] * B[col * K + k];
        }
        C[col * M + row] = acc;
    }
} 


int main(){
    int M = 1024;
    int N = 1024;
    int K = 1024;

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    float* A = (float*)malloc(sizeA);
    float* B = (float*)malloc(sizeB);
    float* C = (float*)malloc(sizeC);

    //initial on CPU
    for(int i = 0;i < M*K;++i) A[i] = 1.0f;
    for(int i = 0;i < K*N;++i) B[i] = 2.0f;

    float* dev_A,*dev_B,*dev_C;
    CHECK_CUDA(cudaMalloc((void**)&dev_A,sizeA));
    CHECK_CUDA(cudaMalloc((void**)&dev_B,sizeB));
    CHECK_CUDA(cudaMalloc((void**)&dev_C,sizeC));

    CHECK_CUDA(cudaMemcpy(dev_A,A,sizeA,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dev_B,B,sizeB,cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(32,32);
    dim3 blocksPerGrid((M + threadsPerBlock.x - 1)/threadsPerBlock.x,
                        (N + threadsPerBlock.y - 1) / threadsPerBlock.y
);   


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    kernel1<<<blocksPerGrid, threadsPerBlock>>>(M, N, K, dev_A, dev_B, dev_C);
    cudaEventRecord(stop);
    

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    CHECK_CUDA(cudaMemcpy(C, dev_C, sizeC, cudaMemcpyDeviceToHost));

    bool success = true;
    float expectedValue = 2.0f * K;
    for (int i = 0; i < M * N; i++) {
        if (C[i] != expectedValue) {
            printf(" 验证失败！索引 %d 处的值为 %f，期望值为 %f\n", i, C[i], expectedValue);
            success = false;
            break;
        }
    }


    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dev_A); cudaFree(dev_B); cudaFree(dev_C);
    free(A); free(B); free(C);
    return 0;
}


