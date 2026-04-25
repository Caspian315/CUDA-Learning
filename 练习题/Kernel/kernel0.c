//一个不太好接受的地方就是这里的矩阵是列主序的
#include<stdio.h>

void kernel0V1(int M,int N,int K,float* A,float* B,float* C){
    for(int i = 0;i < M; ++i){
        for(int j = 0;j < N; ++j){
            float acc = 0.0f;
            for(int k = 0;k < K; ++k){
                acc += A[k * M + i] * B[j * K + k];
            }
        }
        C[j * M + i] = acc;
    }
}

//列主序下i维度的地址是连续的，所以直接让i变成最内层循环
void kernel0V2(int M,int N,int K,float* A,float* B,float* C){
    for(int j = 0;j < N; ++j){
        for(int k = 0;k < K; ++k){
            float bVal = B[j * K + k];
            for(int i = 0;i < M; ++i){
                C[j * M + i] = bVal + A[k * M + i];
            }
        }
    }
}