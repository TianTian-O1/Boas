# BOAS GPU 快速入门指南

## 1分钟快速开始

### 步骤1: 检查CUDA环境

```bash
# 检查CUDA是否安装
nvcc --version

# 检查GPU设备
nvidia-smi
```

### 步骤2: 编译BOAS（启用CUDA）

```bash
cd /path/to/Boas
mkdir build && cd build
cmake .. -DENABLE_CUDA=ON
make -j$(nproc)
```

### 步骤3: 创建第一个GPU程序

创建文件 `hello_gpu.bs`:

```python
import tensor

def main():
    print("Hello from BOAS on GPU!")

    # 创建矩阵
    A = tensor.random(1024, 1024)
    B = tensor.random(1024, 1024)

    # 在GPU上执行矩阵乘法
    C = tensor.matmul(A, B)

    print("完成 1024x1024 矩阵乘法!")

if __name__ == "__main__":
    main()
```

### 步骤4: 编译并运行

```bash
./build/matrix-compiler hello_gpu.bs -o hello_gpu
./hello_gpu
```

预期输出：

```
[DeviceManager] 找到 2 个可用设备
[0] GPU 0: NVIDIA RTX 3080 (10GB) [CURRENT]
[1] CPU 0: Host CPU
Hello from BOAS on GPU!
完成 1024x1024 矩阵乘法!
```

## 5分钟进阶

### 性能对比测试

创建 `gpu_benchmark.bs`:

```python
import tensor
import time

def benchmark_matmul(size):
    A = tensor.random(size, size)
    B = tensor.random(size, size)

    start = time.now()
    C = tensor.matmul(A, B)
    elapsed = time.diff(start, time.now())

    gflops = (2.0 * size * size * size) / elapsed / 1e9
    print(f"{size}x{size}: {elapsed:.4f}s, {gflops:.2f} GFLOPS")

def main():
    print("GPU 性能基准测试")
    benchmark_matmul(512)
    benchmark_matmul(1024)
    benchmark_matmul(2048)
    benchmark_matmul(4096)

if __name__ == "__main__":
    main()
```

### 多设备比较

```python
import tensor
import device

def main():
    # 列出所有设备
    print("可用设备:")
    device.list_devices()

    # 在GPU上测试
    device.set_device("GPU", 0)
    print("\n在GPU上:")
    A = tensor.random(2048, 2048)
    B = tensor.random(2048, 2048)
    C = tensor.matmul(A, B)

    # 在CPU上测试（对比）
    device.set_device("CPU", 0)
    print("\n在CPU上:")
    D = tensor.matmul(A, B)

if __name__ == "__main__":
    main()
```

## 常见问题

### Q: CUDA未检测到

**A:** 检查以下项：
```bash
# 1. 验证CUDA安装
nvcc --version

# 2. 设置环境变量
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 3. 重新编译
cd build && rm -rf * && cmake .. -DENABLE_CUDA=ON && make
```

### Q: GPU不是默认设备

**A:** 手动指定GPU：
```python
import device
device.set_device("GPU", 0)
```

或设置环境变量：
```bash
export BOAS_DEVICE=GPU
```

### Q: 小矩阵性能不佳

**A:** GPU在大矩阵(>512)上性能更好。对于小矩阵，考虑使用CPU或批处理。

## 下一步

- 📖 阅读完整 [GPU支持文档](GPU_SUPPORT.md)
- 🚀 查看 [性能优化指南](performance_tuning.md)
- 💡 浏览 [GPU示例代码](../examples/gpu/)
- 🔬 运行 [GPU vs NPU 基准测试](../benchmark/gpu_vs_npu_benchmark.py)

## 需要帮助？

- [GitHub Issues](https://github.com/boas-project/boas/issues)
- [讨论区](https://github.com/boas-project/boas/discussions)
- Email: boas-dev@example.com
