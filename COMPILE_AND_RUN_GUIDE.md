# BOAS 编译器编译和使用指南

## 🚀 快速开始

### 方法一：一键编译和测试（推荐）

```bash
# 运行快速构建脚本
./quick_build_and_test.sh
```

这个脚本会自动：
1. 设置环境变量
2. 编译 BOAS 编译器
3. 创建测试文件
4. 运行矩阵乘法测试

### 方法二：手动编译步骤

## 📦 前置要求

1. **LLVM 20.0** （已安装在 `/root/llvm-project/llvm_build`）
2. **CMake 3.20+**
3. **GCC 9.0+** 或 **Clang 14+**
4. **Python 3.8+**

## 🔧 编译 BOAS 编译器

### 步骤 1：设置环境变量

```bash
# 设置 LLVM 路径
export LLVM_INSTALL_PATH="/root/llvm-project/llvm_build"
export PATH="$LLVM_INSTALL_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$LLVM_INSTALL_PATH/lib:$LD_LIBRARY_PATH"

# 验证 LLVM 安装
llvm-config --version  # 应该显示 20.0.0
```

### 步骤 2：创建构建目录

```bash
cd /root/Boas/Boas-linux
mkdir -p build
cd build
```

### 步骤 3：配置 CMake

```bash
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_DIR="$LLVM_INSTALL_PATH/lib/cmake/llvm" \
    -DMLIR_DIR="$LLVM_INSTALL_PATH/lib/cmake/mlir" \
    -DCMAKE_CXX_STANDARD=17
```

### 步骤 4：编译

```bash
# 编译 BOAS（使用所有CPU核心）
make -j$(nproc)

# 或者指定核心数
make -j4
```

编译成功后，会在 `build` 目录生成 `boas-compiler` 可执行文件。

## 💻 使用 BOAS 编译器

### 基本用法

```bash
# 编译 BOAS 源文件
./build/boas-compiler <源文件.bs> -o <输出文件>

# 运行编译后的程序
./<输出文件>
```

### 示例 1：Hello World

创建文件 `hello.bs`：

```python
# hello.bs
def main():
    print("Hello, BOAS!")
    a = 10
    b = 20
    print(f"10 + 20 = {a + b}")

if __name__ == "__main__":
    main()
```

编译和运行：

```bash
./build/boas-compiler hello.bs -o hello
./hello
```

### 示例 2：矩阵乘法

创建文件 `matmul.bs`：

```python
# matmul.bs
import tensor

def main():
    # 创建 2x2 矩阵
    A = tensor.create(2, 2, [1, 2, 3, 4])  # [[1, 2], [3, 4]]
    B = tensor.create(2, 2, [5, 6, 7, 8])  # [[5, 6], [7, 8]]
    
    # 矩阵乘法
    C = tensor.matmul(A, B)
    
    print("Matrix A:")
    print(A)
    print("\nMatrix B:")
    print(B)
    print("\nA × B =")
    print(C)
    # 结果应该是 [[19, 22], [43, 50]]

if __name__ == "__main__":
    main()
```

编译和运行：

```bash
./build/boas-compiler matmul.bs -o matmul
./matmul
```

### 示例 3：性能测试

创建文件 `benchmark.bs`：

```python
# benchmark.bs
import tensor
import time

def benchmark_matmul(size):
    """测试指定大小的矩阵乘法性能"""
    A = tensor.random(size, size)
    B = tensor.random(size, size)
    
    start = time.time()
    C = tensor.matmul(A, B)
    end = time.time()
    
    elapsed = end - start
    gflops = (2.0 * size * size * size) / (elapsed * 1e9)
    
    print(f"Size {size}x{size}: {elapsed:.4f}s, {gflops:.2f} GFLOPS")
    return C

def main():
    print("BOAS 矩阵乘法性能测试")
    print("-" * 40)
    
    sizes = [64, 128, 256, 512, 1024]
    for size in sizes:
        benchmark_matmul(size)
    
    print("\n测试完成!")

if __name__ == "__main__":
    main()
```

编译和运行：

```bash
./build/boas-compiler benchmark.bs -o benchmark
./benchmark
```

## 🎯 编译器选项

```bash
# 基本编译
./build/boas-compiler input.bs -o output

# 开启优化
./build/boas-compiler input.bs -o output -O3

# 生成 LLVM IR（用于调试）
./build/boas-compiler input.bs -o output --emit-llvm

# 生成 MLIR（用于调试）
./build/boas-compiler input.bs -o output --emit-mlir

# 针对 NPU 优化
./build/boas-compiler input.bs -o output --target=npu

# 显示帮助
./build/boas-compiler --help
```

## 🐛 常见问题

### 1. 找不到 LLVM

```bash
# 确保设置了环境变量
export LLVM_INSTALL_PATH="/root/llvm-project/llvm_build"
export PATH="$LLVM_INSTALL_PATH/bin:$PATH"
```

### 2. 编译错误：缺少 MLIR

```bash
# 重新配置 CMake，确保 MLIR 路径正确
cmake .. -DMLIR_DIR="$LLVM_INSTALL_PATH/lib/cmake/mlir"
```

### 3. 运行时错误：找不到库

```bash
# 设置库路径
export LD_LIBRARY_PATH="$LLVM_INSTALL_PATH/lib:./build/lib:$LD_LIBRARY_PATH"
```

### 4. 性能不佳

确保使用 Release 模式编译：
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release
```

## 📊 性能优化提示

1. **使用大矩阵**：BOAS 对大矩阵（>512x512）优化更好
2. **批量操作**：将多个小操作合并为批量操作
3. **避免频繁内存分配**：重用矩阵对象
4. **使用 NPU 模式**：编译时加 `--target=npu`

## 🔍 调试技巧

```bash
# 查看生成的 MLIR
./build/boas-compiler test.bs -o test --emit-mlir > test.mlir

# 查看生成的 LLVM IR
./build/boas-compiler test.bs -o test --emit-llvm > test.ll

# 使用 gdb 调试
gdb ./test
```

## 📚 更多示例

查看 `examples/` 目录：
- `hello_world.bs` - 入门示例
- `matrix_ops.bs` - 各种矩阵运算
- `npu_optimized.bs` - NPU 优化示例

## ✅ 验证安装

运行以下命令验证 BOAS 是否正确安装：

```bash
# 编译器版本
./build/boas-compiler --version

# 运行测试套件
cd build
make test

# 运行示例
./build/boas-compiler ../examples/hello_world.bs -o hello_test
./hello_test
```

如果一切正常，你应该看到：
- 编译器版本信息
- 测试全部通过
- Hello World 输出

## 🎉 开始使用 BOAS！

现在你可以：
1. 编写 Python 风格的 `.bs` 文件
2. 使用 `boas-compiler` 编译
3. 享受超越 C++ 的性能！

祝你使用愉快！如有问题，请查看文档或提交 Issue。