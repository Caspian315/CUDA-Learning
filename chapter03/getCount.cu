#include "../common/book.h"

int main(){
    cudaDeviceProp prop;

    int count;
    HANDLE_ERROR(cudaGetDeviceCount(&count));
    for(int i = 0;i < count; ++i){
        HANDLE_ERROR(cudaGetDeviceProperties(&prop,i));
        printf("--- 第 %d 张显卡 (Device %d) 核心信息 ---\n", i, i);
        
        // 1. 显卡真名
        printf("显卡名称 (Name): %s\n", prop.name);
        
        // 2. 算力代号 (Compute Capability)
        // 这决定了你的卡支持哪些高级硬件特性（比如 Tensor Cores）
        printf("计算能力 (Compute Capability): %d.%d\n", prop.major, prop.minor);
        
        // 3. 显存大小（从字节换算成 MB）
        // 这是你未来跑大模型或者大矩阵的物理容量上限
        printf("全局显存总量 (Total Global Memory): %lu MB\n", prop.totalGlobalMem / (1024 * 1024));
        
        // 4. 流多处理器数量 (SM)
        // 这是 GPU 真正的“干活车间”，数量越多，并发能力越恐怖
        printf("流多处理器数量 (Multiprocessor Count): %d\n", prop.multiProcessorCount);
        
        // 5. 线程排布的物理极限
        // 决定了你写 <<<grid, block>>> 时，block 里的数字最大能填多少
        printf("每个线程块最大线程数 (Max Threads per Block): %d\n", prop.maxThreadsPerBlock);
        
        printf("----------------------------------------\n\n");
    }

    return 0;
}