# BOAS GPU Support Documentation

## 概述

BOAS v1.1.0 新增了对 Nvidia GPU (CUDA) 的支持，现在可以在以下设备上运行：

- ✅ **Nvidia GPU** (通过 CUDA)
- ✅ **华为昇腾 NPU** (通过 CANN)
- ✅ **CPU** (作为后备)

系统会自动检测并选择最佳可用设备。

## 环境要求

### GPU支持
- Nvidia GPU (Compute Capability 6.0+)
  - Pascal (GTX 1000系列)
  - Volta (V100, Titan V)
  - Turing (RTX 2000系列)
  - Ampere (RTX 3000/4000系列, A100)
  - Ada/Hopper (RTX 4000系列, H100)
- CUDA Toolkit 11.0+ (推荐 12.0+)
- cuBLAS库

### NPU支持
- 华为昇腾 NPU (910A/910B/310P)
- CANN Toolkit 6.0+

## 编译配置

### 启用CUDA支持（默认开启）

```bash
mkdir build && cd build
cmake .. -DENABLE_CUDA=ON
make -j$(nproc)
```

### 禁用CUDA支持

```bash
cmake .. -DENABLE_CUDA=OFF
make -j$(nproc)
```

### 查看编译配置

```bash
cmake .. -DENABLE_CUDA=ON
# 输出：
# -- CUDA Toolkit found: 12.0
# -- CUDA Include dirs: /usr/local/cuda/include
# -- CUDA support enabled
# -- Found CANN toolkit at: /usr/local/Ascend/ascend-toolkit/latest
```

## 设备选择

### 自动选择（推荐）

BOAS会自动检测并选择最佳设备，优先级：**GPU > NPU > CPU**

```python
import tensor

def main():
    # 自动使用最佳设备
    A = tensor.random(1024, 1024)
    B = tensor.random(1024, 1024)
    C = tensor.matmul(A, B)  # 自动在GPU/NPU上执行
```

### 手动指定设备

可以通过环境变量指定设备：

```bash
# 使用GPU
export BOAS_DEVICE=GPU
./my_program

# 使用NPU
export BOAS_DEVICE=NPU
./my_program

# 使用CPU
export BOAS_DEVICE=CPU
./my_program
```

或在代码中指定：

```python
import tensor
import device

def main():
    # 列出所有可用设备
    device.list_devices()

    # 设置为GPU
    device.set_device("GPU", 0)

    # 或设置为NPU
    device.set_device("NPU", 0)
```

## GPU优化特性

### 1. 多种矩阵乘法内核

BOAS根据矩阵大小自动选择最优kernel：

- **小矩阵** (< 512): 基础kernel，低延迟
- **中等矩阵** (512-2048): 共享内存优化，平衡性能
- **大矩阵** (> 2048): 高级优化kernel，最大吞吐量

### 2. cuBLAS集成

对于标准矩阵乘法，使用高度优化的cuBLAS库：

```cpp
// 自动使用cuBLAS
bool success = cudaRuntime.executeMatmul(A, B, C, M, K, N);
```

### 3. 自定义CUDA Kernel

支持自定义CUDA kernel优化：

```cuda
// 使用共享内存的分块矩阵乘法
__global__ void matmul_kernel_shared<32>(
    const float* A, const float* B, float* C,
    int M, int N, int K
) {
    __shared__ float As[32][32];
    __shared__ float Bs[32][32];
    // ... 优化实现
}
```

### 4. 性能优化

- **Tensor Core支持** (Volta+架构)
- **多GPU支持** (可选择不同GPU设备)
- **异步执行** (与CPU计算重叠)
- **内存池管理** (减少分配开销)

## 性能对比

### GPU vs NPU 基准测试

运行性能对比测试：

```bash
python3 benchmark/gpu_vs_npu_benchmark.py
```

预期性能（示例）：

| 矩阵大小 | RTX 3080 | Ascend 910A | PyTorch GPU |
|---------|----------|-------------|-------------|
| 1024x1024 | 1.2 ms | 1.5 ms | 2.1 ms |
| 2048x2048 | 6.8 ms | 8.2 ms | 11.5 ms |
| 4096x4096 | 45 ms | 58 ms | 75 ms |

### C++ 测试程序

```bash
# 编译测试程序
cd build
cmake ..
make

# 运行GPU测试
./tests/gpu/test_cuda_runtime

# 输出：
# === Available Devices ===
# [0] GPU 0: NVIDIA GeForce RTX 3080 (Compute 8.6, 10240 MB, 68 SMs) [CURRENT]
# [1] NPU 0: Ascend 910A
# [2] CPU 0: Host CPU (fallback)
```

## 示例代码

### GPU矩阵乘法

```python
# test_gpu_matmul.bs
import tensor

def main():
    print("GPU 矩阵乘法测试")

    # 小矩阵
    A = tensor.create(4, 4, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    B = tensor.create(4, 4, [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
    C = tensor.matmul(A, B)
    print("Result:", C)

    # 大矩阵
    A_large = tensor.random(2048, 2048)
    B_large = tensor.random(2048, 2048)
    C_large = tensor.matmul(A_large, B_large)
    print("完成 2048x2048 矩阵乘法")

if __name__ == "__main__":
    main()
```

编译并运行：

```bash
./build/matrix-compiler test_gpu_matmul.bs -o test_gpu
./test_gpu
```

### 设备管理

```python
import tensor
import device

def main():
    # 查看所有设备
    devices = device.list_devices()
    print(f"找到 {len(devices)} 个设备")

    for dev in devices:
        print(f"  - {dev.name}: {dev.properties}")

    # 选择GPU
    if device.is_available("GPU"):
        device.set_device("GPU", 0)
        print("使用 GPU 设备")

        # 在GPU上执行计算
        A = tensor.random(1024, 1024)
        B = tensor.random(1024, 1024)
        C = tensor.matmul(A, B)
    else:
        print("GPU 不可用，使用 CPU")
```

## API 参考

### DeviceManager

```cpp
#include "mlirops/DeviceManager.h"

// 获取设备管理器实例
auto& deviceMgr = DeviceManager::getInstance();

// 初始化（自动检测所有设备）
deviceMgr.initialize();

// 列出可用设备
auto devices = deviceMgr.getAvailableDevices();

// 设置设备
deviceMgr.setDevice(DeviceType::GPU, 0);

// 自动选择最佳设备
deviceMgr.selectBestDevice();
```

### CUDARuntime

```cpp
#include "mlirops/CUDARuntime.h"

// 获取CUDA运行时实例
auto& cudaRuntime = CUDARuntime::getInstance();

// 初始化
cudaRuntime.initialize();

// 获取设备数量
int count = cudaRuntime.getDeviceCount();

// 执行矩阵乘法
cudaRuntime.executeMatmul(A, B, C, M, K, N);
```

### GPUBackend

```cpp
#include "mlirops/GPUBackend.h"

// 初始化GPU后端
GPUBackend::initialize();

// 检查GPU可用性
if (GPUBackend::isAvailable()) {
    // 生成GPU优化的MLIR
    auto result = GPUBackend::generateGPUMatmul(
        generator, lhs, rhs, M, N, K
    );
}
```

## 故障排除

### CUDA未找到

```
错误: CUDA Toolkit not found, GPU support will be disabled
```

解决方案：
1. 安装CUDA Toolkit：https://developer.nvidia.com/cuda-downloads
2. 设置环境变量：
   ```bash
   export CUDA_HOME=/usr/local/cuda
   export PATH=$CUDA_HOME/bin:$PATH
   export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
   ```

### GPU设备未检测到

```
错误: No CUDA-capable GPU devices found
```

解决方案：
1. 检查GPU驱动：`nvidia-smi`
2. 验证CUDA安装：`nvcc --version`
3. 检查GPU可见性：`echo $CUDA_VISIBLE_DEVICES`

### cuBLAS错误

```
错误: cublasSgemm failed: CUBLAS_STATUS_NOT_INITIALIZED
```

解决方案：
1. 确保cuBLAS已正确安装
2. 检查CUDA库路径是否正确
3. 重新初始化CUDA运行时

## 高级配置

### 环境变量

- `BOAS_DEVICE`: 指定设备类型 (GPU/NPU/CPU)
- `CUDA_VISIBLE_DEVICES`: 限制可见的GPU设备
- `BOAS_GPU_BLOCK_SIZE`: 自定义GPU block大小

### CMake选项

```bash
# 指定CUDA架构
cmake .. -DCMAKE_CUDA_ARCHITECTURES="75;80;86"

# 禁用CUDA
cmake .. -DENABLE_CUDA=OFF

# 使用特定CUDA路径
cmake .. -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.0
```

## 性能调优

### 1. 选择合适的矩阵大小

GPU在大矩阵上性能更好（>= 512），小矩阵可能CPU更快。

### 2. 批处理

多个小矩阵操作可以批处理以提高GPU利用率。

### 3. 数据传输优化

最小化CPU-GPU数据传输，尽量在GPU上完成所有计算。

### 4. 使用合适的数据类型

- FP16：更快，但精度较低
- FP32：平衡
- FP64：最高精度，但速度较慢

## 未来计划

- [ ] FP16/BF16混合精度支持
- [ ] 多GPU并行执行
- [ ] Tensor Core专用优化
- [ ] AMD GPU (ROCm) 支持
- [ ] Intel GPU (oneAPI) 支持

## 相关文档

- [CUDA编程指南](https://docs.nvidia.com/cuda/)
- [cuBLAS文档](https://docs.nvidia.com/cuda/cublas/)
- [BOAS架构设计](docs/architecture.md)
- [性能优化指南](docs/performance.md)
