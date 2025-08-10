# Boas语言昇腾NPU适配指南

## 概述

本文档介绍了Boas编程语言如何适配昇腾NPU设备，实现高性能矩阵运算。此适配基于昇腾CANN工具链和triton-ascend项目的设计思路。

## 适配架构

### 1. 整体架构

```
Boas语言层 (tensor.matmul)
    ↓
AST层 (MatmulExprAST) 
    ↓
MLIR生成层 (MLIRGenMatrix.cpp)
    ↓
NPU后端 (NPUBackend.cpp)
    ↓
Triton Kernel生成 (NPUTritonGenerator.cpp)
    ↓
昇腾CANN编译器
    ↓
NPU二进制执行
```

### 2. 核心组件

#### 2.1 NPU后端 (`NPUBackend.h/cpp`)
- **功能**: NPU设备管理和优化策略选择
- **特性**:
  - NPU设备检测和初始化
  - 矩阵乘法优化策略选择
  - 与现有MLIR基础设施集成

#### 2.2 NPU环境检测 (`NPUEnvironment.cpp`)
- **功能**: 运行时NPU环境检测
- **检测项目**:
  - CANN工具链安装
  - NPU设备可用性
  - torch_npu库支持

#### 2.3 Triton Kernel生成器 (`NPUTritonGenerator.cpp`) 
- **功能**: 生成NPU优化的Triton kernel代码
- **优化特性**:
  - 8x8对角线分核策略
  - 512B对齐优化
  - Bank冲突避免
  - L2 Cache优化

## NPU优化策略

### 1. 对角线分核 (Diagonal Tiling)

当矩阵在M和N方向均超过8块时，启用8x8对角线分核策略:

**优势**:
- 减少Bank冲突
- 提升L2 Cache命中率
- 优化内存访问模式

**适用场景**:
- 大矩阵乘法 (>= 8x8 blocks)
- 右矩阵大小超过L2 Cache容量

### 2. 块配置优化

**推荐配置** (针对昇腾NPU 512B对齐优化):
- BLOCK_M = 128
- BLOCK_N = 256  
- BLOCK_K = 256

**自适应策略**:
- 根据矩阵大小动态选择块配置
- 考虑AI Core数量进行负载均衡

### 3. 内存优化

- **FP32累加器**: 提高计算精度
- **BF16存储**: 减少内存带宽需求
- **向量化访问**: 提升内存吞吐

## 使用方法

### 1. 环境准备

```bash
# 1. 运行环境配置脚本
./scripts/setup_npu_env.sh

# 2. 检查NPU环境
python3 test_npu_basic.py
```

### 2. 编译Boas项目

```bash
# 编译包含NPU支持的Boas
mkdir build && cd build
cmake .. -DLLVM_INSTALL_PREFIX=/path/to/llvm-install
make -j$(nproc)
```

### 3. 运行NPU测试

```bash
# 运行NPU矩阵乘法测试
./run.sh -d all -t test/test_npu_matmul.bs
```

### 4. Boas代码示例

```python
import tensor

def main():
    # 创建矩阵 - 自动检测并使用NPU
    A = tensor.random(1024, 1024)
    B = tensor.random(1024, 1024)
    
    # 矩阵乘法 - 自动使用NPU优化
    C = tensor.matmul(A, B)
    
    print("NPU矩阵乘法完成")
```

## 技术细节

### 1. NPU检测逻辑

```cpp
bool NPUBackend::isAvailable() {
    // 1. 检查CANN环境
    // 2. 检查NPU设备文件
    // 3. 检查torch_npu可用性
    return NPUEnvironment::checkNPUEnvironment();
}
```

### 2. 优化策略选择

```cpp
mlir::Value generateMLIRForMatmul(const MatmulExprAST* expr) {
    // 检查NPU可用性
    if (NPUBackend::isAvailable()) {
        // 使用NPU优化路径
        return NPUBackend::generateNPUMatmul(this, lhs, rhs, m, n, k);
    } else {
        // 回退到CPU优化路径
        return createOptimizedMatmul(lhs, rhs, result, m, n, k);
    }
}
```

### 3. Triton Kernel生成

生成的kernel特点:
- 支持动态形状
- NPU特定优化
- 高精度计算
- 内存访问优化

## 性能优化要点

### 1. 编译时优化
- 块大小自适应选择
- 循环展开和向量化
- 内存访问模式优化

### 2. 运行时优化  
- AI Core负载均衡
- 内存对齐和预取
- 计算与内存访问重叠

### 3. 算法优化
- 对角线分核减少冲突
- L2 Cache友好的访问模式
- 精度和性能平衡

## 调试和性能分析

### 1. 启用调试信息

```bash
# 设置环境变量启用详细日志
export ASCEND_SLOG_PRINT_TO_STDOUT=1
export ASCEND_GLOBAL_LOG_LEVEL=0

# 运行程序并保存日志
./matrix-compiler --run test/test_npu_matmul.bs 2>&1 | tee npu_debug.log
```

### 2. 性能分析

```bash
# 使用NPU profiler
npu-smi info  # 查看设备状态
# 运行性能测试
./benchmark/scripts/run_bench.sh
```

## 故障排除

### 1. 常见问题

**NPU设备不可用**:
- 检查驱动安装: `ls /dev/davinci*`
- 检查CANN环境: `echo $ASCEND_HOME`
- 验证设备权限: `ls -la /dev/davinci*`

**编译错误**:
- 确保LLVM/MLIR版本兼容
- 检查头文件路径
- 验证CMake配置

**运行时错误**:
- 检查torch_npu版本兼容性
- 验证内存配置
- 查看NPU日志

### 2. 性能问题

**性能不达预期**:
- 检查块配置是否适合矩阵大小
- 验证是否启用了对角线分核
- 分析内存访问模式

**内存不足**:
- 减小块大小配置
- 使用更低精度数据类型
- 优化内存池使用

## 扩展开发

### 1. 添加新的优化策略

继承`NPUMatmulOptimizer`类，实现新的配置策略:

```cpp
class CustomNPUOptimizer : public NPUMatmulOptimizer {
public:
    static BlockConfig getCustomConfig(int M, int N, int K) {
        // 自定义优化逻辑
    }
};
```

### 2. 支持其他NPU操作

参考矩阵乘法的实现模式，添加其他算子的NPU支持:
- 卷积操作
- 激活函数
- 归一化操作

## 参考资料

1. [昇腾CANN开发指南](https://www.hiascend.com/document)
2. [triton-ascend项目](https://gitee.com/ascend/triton-ascend)
3. [Triton编程指南](https://triton-lang.org/)
4. [MLIR官方文档](https://mlir.llvm.org/)

## 版本历史

- v1.0: 基础NPU矩阵乘法支持
- v1.1: 添加对角线分核优化
- v1.2: 支持动态形状和自适应配置

---

**注意**: 此适配当前处于开发阶段，建议在生产环境使用前进行充分测试。
