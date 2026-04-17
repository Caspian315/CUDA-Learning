// 实际上，第一个版本的矢量加法存在问题
// 假如说N=100000,单纯的增加block数量是一种及其浪费硬件的做法
// 因此就可以充分利用Block内部的线程并发
#include<stdio.h>
#include<stdlib.h>
#include "../common/book.h"

#define N 100000

__global__ void add(int *a,int *b,int *c,int n){
    // 1. 获取当前线程的全局绝对 ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // 2. 计算整个网格一次能提供的最大线程总数（跨步长度）
    int stride = blockDim.x * gridDim.x;

    // 3. 只要当前索引没有越界，就算完一个往后跳 stride 步继续算
    for(int i = tid;i < N; i += stride){
        c[i] = a[i] + b[i];
    }
}

int main(void){
    int *a = (int*)malloc(N * sizeof(int));
    int *b = (int*)malloc(N * sizeof(int));
    int *c = (int*)malloc(N * sizeof(int));

    int *dev_a, *dev_b, *dev_c;

    HANDLE_ERROR(cudaMalloc((void**)&dev_a,N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b,N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_c,N * sizeof(int)));

    // 在 CPU 上初始化十万个数据
    for(int i = 0;i < N; ++i){
        a[i] = i;
        b[i] = i * i;
    }

    HANDLE_ERROR(cudaMemcpy(dev_a,a,N * sizeof(int),cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b,b,N * sizeof(int),cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocksPerGrid = 32;

    add<<<blocksPerGrid,threadsPerBlock>>> (dev_a,dev_b,dev_c,N);
    HANDLE_ERROR(cudaMemcpy(c,dev_c,N * sizeof(int),cudaMemcpyDeviceToHost));

    //  验证结果（挑几个抽查，避免刷屏）
    bool success = true;
    for(int i = 0; i < N; ++i){
        if (c[i] != a[i] + b[i]){
            printf("Error: 在索引 %d 处计算错误！ %d + %d != %d\n", i, a[i], b[i], c[i]);
            success = false;
            break;
        }
    }
    
    if(success){
        printf("牛逼！100,000 个元素的并行矢量加法完美通过测试！\n");
        printf("抽查最后一个元素：a[%d] + b[%d] = %d + %d = %d\n", 
               N-1, N-1, a[N-1], b[N-1], c[N-1]);
    }

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    free(a);
    free(b);
    free(c);

    return 0;
}