# CUDA编程学习笔记



## Chapter2 配置环境

这里选用在wsl2中搭建ubuntu子系统，版本号为22.02

**1. 准备工作** 打开终端（Terminal），先更新一下软件列表：

Bash

```
sudo apt update
sudo apt upgrade
```

**2. 查找推荐驱动** 输入以下命令，系统会列出你的显卡型号以及所有可用的驱动版本：

Bash

```
ubuntu-drivers devices
```

在输出的列表中，寻找带有 `recommended` 字样的那一行（例如 `nvidia-driver-535 - distro non-free recommended`）。记住这个数字 `535`（你的电脑上可能是其他数字）。

**3. 执行安装** 使用 `apt` 命令安装那个被推荐的版本（将 `535` 替换为你看到的版本号）：

Bash

```
sudo apt install nvidia-driver-535
```

**4. 重启系统** 安装完成后，必须重启电脑让驱动内核模块生效：

Bash

```
sudo reboot
```

配置成功



## Chapter3 CUDA C 简介

**1. 异构计算核心概念**

- **Host (主机)**：指代 CPU 及其系统内存（DDR）。
- **Device (设备)**：指代 GPU 及其显存（VRAM）。
- **物理隔离准则**：Host 和 Device 拥有完全独立的内存空间。**绝对不能在 Host 代码中对 Device 指针解引用**（会引发段错误），必须通过专用的总线通信函数（如 `cudaMemcpy`）进行数据搬运。

**2. 核心语法与编译**

- **核函数 (Kernel)**：使用 `__global__` 修饰符声明。这类函数由 CPU (Host) 发起调用，但在 GPU (Device) 上执行。
- **执行配置 (Execution Configuration)**：在调用核函数时使用的特殊语法 `<<<Grid, Block>>>`。
  - 示例：`add<<<1, 1>>>(a, b, dev_c);` 表示启动 1 个线程块（Block），该块内包含 1 个线程（Thread）。
- **编译命令**：使用 NVIDIA 编译器驱动 `nvcc`。
  - Bash: `nvcc -o hello hello.cu`

**3. 内存管理三大件 (API)**

- **分配设备内存**：`cudaMalloc`
  - 语法：`cudaMalloc((void**)&dev_ptr, sizeof(int));`
  - *避坑笔记*：必须按址传递（传入双重指针 `&dev_ptr`），并且强制转换为无类型指针 `(void**)`，以便 API 能将分配好的 GPU 内存物理地址写回该变量。
- **数据拷贝**：`cudaMemcpy`
  - 语法：`cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);`
  - 常用的方向参数：`cudaMemcpyDeviceToHost` (设备到主机) 和 `cudaMemcpyHostToDevice` (主机到设备)。
- **释放内存**：`cudaFree`
  - 语法：`cudaFree(dev_ptr);`

**4. 设备查询与属性** 在编写复杂算法前，通常需要动态查询显卡硬件指标，以防止爆显存或超出物理线程极限。

- **查询可用显卡数量**：

  C

  ```
  int count;
  cudaGetDeviceCount(&count);
  ```

- **查询显卡详细属性（硬件户口本）**：

  C

  ```cu
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device_id);
  // 关键属性：
  // prop.major / minor (计算能力架构)
  // prop.totalGlobalMem (全局显存大小)
  // prop.multiProcessorCount (SM 数量)
  // prop.maxThreadsPerBlock (单 Block 最大线程数，通常为 1024)
  ```



## Chapter04 CUDA C并行编程

### Chapter 4 第一个并行程序：矢量求和 (Vector Addition)

本节通过实现两个数组的并行相加，走通了完整的 CUDA 异构计算工作流，并跨越了几个经典的 C/C++ 语法与并行思维陷阱。

#### 1. 完整的异构计算闭环

一个标准的 CUDA 算子调度流程必须严格遵循以下 5 个步骤：

1. **显存分配**：使用 `cudaMalloc` 为输入和输出数据在设备端（GPU）开辟空间。
2. **数据下发 (H2D)**：使用 `cudaMemcpy(..., cudaMemcpyHostToDevice)` 将主机（CPU）初始化的数据拷贝到 GPU。
3. **核函数发射**：通过 `<<<Grid, Block>>>` 语法配置并发规模并调用 `__global__` 函数。
4. **数据回传 (D2H)**：使用 `cudaMemcpy(..., cudaMemcpyDeviceToHost)` 将 GPU 算好的结果拷贝回主机内存。
5. **释放资源**：严格成对使用 `cudaFree` 释放设备指针，防止显存泄漏。

#### 2. CUDA 核心内置变量 (Built-in Variables)

在传统的 C 语言中，我们在 `for` 循环里使用 `i` 来遍历数组。在 GPU 的并行世界里，没有显式的循环，数组的索引由**当前线程的硬件物理坐标**来决定：

- `blockIdx.x`：获取当前线程所在的线程块（Block）在 X 维度上的索引（ID）。
- **并行映射思维**：`int tid = blockIdx.x;`，直接让第 `tid` 个线程，去处理数组中下标为 `tid` 的元素。

#### 3. 执行配置的硬件极限

- 调用语法：`add<<<N, 1>>>` 意味着启动了 $N$ 个线程块（Block），每个块里只有 $1$ 个线程（Thread）。
- **思考题留存**：当前通过增加 Block 的数量来覆盖数组长度 $N$。但由于硬件（如 RTX 5060）存在并发调度的物理上限，当 $N$ 达到百万级别（如 $N = 100,000$）时，这种纯靠拉高 Block 数量的策略将面临硬件限制，需要引入更高级的线程组织策略。

**but!!!**

当我们面临十万、百万级别的大规模数据（例如 $N=100,000$）时，单纯依赖增加 Block 数量（如 `<<<N, 1>>>`）会导致大量硬件资源浪费，甚至超出 GPU 网格维度的物理上限。

为了真正榨干 GPU 的算力，必须充分利用 **Block 内部的线程并发**，并引入工业界标准的**网格跨步循环**模式。

#### 1. 核心思想：让线程“跑起来搬砖”

不要为每一个数据分配一个专属的死板线程。相反，我们只启动足够“塞满”显卡流多处理器（SM）的线程总数。这批线程处理完当前的数据后，集体向后跳跃一个**网格总长度（Stride）**，继续处理下一波数据，直到遍历完整个大数组。

#### 2. 终极寻址与跨步公式

在核函数内部，寻址逻辑从单纯的 `blockIdx.x` 进化为以下三步：

C

```
__global__ void add(int *a, int *b, int *c, int n){
    // 1. 获取当前线程的全局绝对 ID（我是全网格中的第几个工人）
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 2. 计算步长：整个网格一次发射的线程总数
    int stride = gridDim.x * blockDim.x; 

    // 3. 跨步循环：只要没越界，算完一个往后跳 stride 步
    for (int i = tid; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}
```

#### 3. 向上取整的数学魔法

在 CPU 端配置执行参数 `<<<Grid, Block>>>` 时，我们需要计算需要多少个 Block。

如果直接使用整数除法 `N / threadsPerBlock`（向下取整），会导致尾部无法被整除的数据被直接抛弃。

**工业标准写法（向上取整 Ceiling）：**

C

```
int threadsPerBlock = 256;
// 核心公式：(被除数 + 除数 - 1) / 除数
int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock; 

add<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_c, N);
```

*原理：加上 `除数 - 1` 后再做截断除法，能完美保证即便多出 1 个数据，也会为其额外分配一个完整的 Block。多余空闲的线程会由核函数内部的 `i < n` 越界保护挡住，保证安全。*

#### 