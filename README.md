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
