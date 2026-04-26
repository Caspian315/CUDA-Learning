//kernel3引入 WPT
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

#define TS 32
#define WPT 8
#define RTS (TS / WPT)

__global__ void kernel3(int M,int N,int K,float* A,float* B,float* C){
    int row = threadIdx.x;
    int col = threadIdx.y;

    int globalRow = TS * blockIdx.x + row;
    int globalCol = TS * blockIdx.y + col;

    __shared__ float Asub[TS][TS];
    __shared__ float Bsub[TS][TS];

    float acc[WPT] = {0.0f};
    int numTiles = K / TS;

    for(int t = 0;t < numTiles; ++t){
        for(int w = 0;w < WPT; ++w){
        int tileCol = TS * t + col + w * RTS;
        int tileRow = TS * t + row;

        Asub[col + w * RTS][row] = A[tileCol * M + globalRow];
        Bsub[col + w * RTS][row] = B[(globalCol + w * RTS) * K + tileRow];

        }

         __syncthreads();

        for(int k = 0;k < TS; ++k){
            for(int w = 0;w < WPT; ++w){
            acc[w] += Asub[k][row] * Bsub[col + w * RTS][k];
        }
    }
        __syncthreads();
  }
    for(int w = 0; w < WPT; ++w){
        C[(globalCol + w * RTS)*M + globalRow] = acc[w];
    }

}


int main() {
    int M = 1024;
    int N = 1024;
    int K = 1024;

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    float* A = (float*)malloc(sizeA);
    float* B = (float*)malloc(sizeB);
    float* C = (float*)malloc(sizeC);

    for(int i = 0; i < M*K; ++i) A[i] = 1.0f;
    for(int i = 0; i < K*N; ++i) B[i] = 2.0f;

    float *dev_A, *dev_B, *dev_C;
    CHECK_CUDA(cudaMalloc((void**)&dev_A, sizeA));
    CHECK_CUDA(cudaMalloc((void**)&dev_B, sizeB));
    CHECK_CUDA(cudaMalloc((void**)&dev_C, sizeC));

    CHECK_CUDA(cudaMemcpy(dev_A, A, sizeA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dev_B, B, sizeB, cudaMemcpyHostToDevice));

    //线程块变成了 32 x 4
    dim3 threadsPerBlock(TS, RTS); 
    
    // Grid 的大小依然是整体除以 32，因为一个 Block 整体还是负责 32x32 的矩阵区域
    dim3 blocksPerGrid((M + TS - 1) / TS, (N + TS - 1) / TS);


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    kernel3<<<blocksPerGrid, threadsPerBlock>>>(M, N, K, dev_A, dev_B, dev_C);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    CHECK_CUDA(cudaMemcpy(C, dev_C, sizeC, cudaMemcpyDeviceToHost));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dev_A); cudaFree(dev_B); cudaFree(dev_C);
    free(A); free(B); free(C);

    return 0;
}