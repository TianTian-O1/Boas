# Boas MLIR Dialect 完整指南

## 概述

Boas MLIR Dialect是专为Boas编程语言设计的自定义MLIR方言，替代了对Triton的依赖，提供了更好的控制和优化能力。该方言专门针对矩阵运算、张量操作和NPU优化设计。

## 设计理念

### 为什么需要自定义Dialect？

1. **完全控制**：相比依赖外部Triton，自定义dialect提供了完全的控制权
2. **语义匹配**：直接对应Boas语言的高级语义
3. **优化空间**：可以实现Boas特定的优化策略
4. **设备感知**：内置设备信息，支持异构计算
5. **扩展性**：易于扩展新的操作和优化

### 核心特性

- **高级语义操作**：直接表示矩阵乘法、张量创建等高级概念
- **NPU特定优化**：内置昇腾NPU优化策略
- **设备感知类型系统**：类型中包含设备信息
- **自动并行化**：支持自动并行化和kernel生成
- **渐进式Lowering**：通过多个pass逐步lowering到底层

## Dialect结构

### 类型系统

#### 1. TensorType (`!boas.tensor`)

```mlir
!boas.tensor<shape x element_type @ device>
```

**示例**：
- `!boas.tensor<1024x1024xf32@npu>` - NPU上的1024x1024浮点矩阵
- `!boas.tensor<?x?xbf16@npu>` - NPU上的动态形状bfloat16矩阵
- `!boas.tensor<512x256xf64@cpu>` - CPU上的512x256双精度矩阵

**特性**：
- 支持静态和动态形状
- 内置设备信息
- 支持多种数据类型（f32, f64, bf16等）

#### 2. MatrixType (`!boas.matrix`)

```mlir
!boas.matrix<rows x cols x element_type @ device>
```

特化的二维张量类型，专门用于矩阵运算优化。

### 属性系统

#### NPUOptimizationAttr (`#boas.npu_opt`)

```mlir
#boas.npu_opt<block_m, block_n, block_k, use_diagonal_tiling, strategy>
```

**示例**：
```mlir
#boas.npu_opt<128, 256, 256, true, "diagonal">
```

包含NPU特定优化配置：
- **block_m/n/k**：块大小配置
- **use_diagonal_tiling**：是否启用对角线分核
- **strategy**：优化策略（"diagonal", "sequential", "auto"）

## 核心操作

### 1. 矩阵乘法 (`boas.matmul`)

```mlir
%result = boas.matmul %lhs, %rhs {npu_opt = #boas.npu_opt<128,256,256,true,"diagonal">} 
    : (!boas.tensor<1024x512xbf16@npu>, !boas.tensor<512x1024xbf16@npu>) 
    -> !boas.tensor<1024x1024xbf16@npu>
```

**特性**：
- 自动设备检测和优化
- 支持动态形状
- NPU特定优化策略
- 类型安全的形状检查

### 2. 张量创建 (`boas.tensor.create`)

```mlir
%tensor = boas.tensor.create(%rows, %cols, %values) {device = "npu"} 
    : (!boas.tensor<2x2xf32@npu>)
```

### 3. 随机张量 (`boas.tensor.random`)

```mlir
%tensor = boas.tensor.random(%rows, %cols) {device = "npu"} 
    : !boas.tensor<?x?xf32@npu>
```

### 4. NPU Kernel操作

#### NPU Kernel (`boas.npu.kernel`)

```mlir
%results = boas.npu.kernel %inputs -> %outputs 
    kernel "boas_matmul" 
    config #boas.npu_opt<128,256,256,true,"diagonal"> {
    // kernel body
} : functional-type(%inputs, %results)
```

#### NPU Launch (`boas.npu.launch`)

```mlir
boas.npu.launch grid (20, 1, 1) block (1, 1, 1) {
    // launch body
}
```

### 5. 设备操作

#### 设备选择 (`boas.device.select`)

```mlir
boas.device.select "npu"
```

#### 设备转移 (`boas.to_device`)

```mlir
%npu_tensor = boas.to_device %cpu_tensor to "npu" 
    : !boas.tensor<1024x1024xf32@cpu> -> !boas.tensor<1024x1024xf32@npu>
```

## Lowering Passes

### 1. Boas到Linalg Lowering (`boas-to-linalg`)

将Boas的高级操作转换为Linalg结构化操作：

```mlir
// 输入
%c = boas.matmul %a, %b : (!boas.tensor<M x K x f32@npu>, !boas.tensor<K x N x f32@npu>) 
                       -> !boas.tensor<M x N x f32@npu>

// 输出
%empty = tensor.empty [%M, %N] : tensor<?x?xf32>
%zero = arith.constant 0.0 : f32
%init = linalg.fill ins(%zero : f32) outs(%empty : tensor<?x?xf32>) -> tensor<?x?xf32>
%result = linalg.matmul ins(%a, %b : tensor<?x?xf32>, tensor<?x?xf32>) 
                       outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>
```

### 2. NPU优化Pass (`boas-npu-opt`)

自动添加NPU优化配置：

**优化策略**：
- 分析矩阵大小
- 选择最优块配置
- 决定是否使用对角线分核
- 添加内存优化提示

**示例转换**：
```mlir
// 输入：无优化配置
%c = boas.matmul %a, %b

// 输出：自动添加NPU优化
%c = boas.matmul %a, %b {npu_opt = #boas.npu_opt<128,256,256,true,"diagonal">}
```

### 3. 设备感知优化Pass (`boas-device-opt`)

- 自动设备选择
- 设备间数据传输优化
- 异构计算调度

### 4. NPU Kernel生成Pass (`npu-kernel-gen`)

将优化后的操作转换为NPU kernel调用：

```mlir
// 输入：Boas操作
%c = boas.matmul %a, %b {npu_opt = #boas.npu_opt<...>}

// 输出：NPU kernel调用
%c = boas.npu.kernel %a, %b -> %c 
    kernel "boas_matmul_1024x1024_diagonal" 
    config #boas.npu_opt<128,256,256,true,"diagonal"> {
    // 生成的kernel body
}
```

## 使用示例

### 基本矩阵乘法

**Boas源码**：
```python
A = tensor.random(1024, 1024)
B = tensor.random(1024, 1024)
C = tensor.matmul(A, B)
```

**生成的Boas Dialect MLIR**：
```mlir
func.func @main() {
  %rows = arith.constant 1024 : index
  %cols = arith.constant 1024 : index
  
  // 创建随机张量
  %A = boas.tensor.random(%rows, %cols) {device = "npu"} 
      : !boas.tensor<1024x1024xf32@npu>
  %B = boas.tensor.random(%rows, %cols) {device = "npu"}
      : !boas.tensor<1024x1024xf32@npu>
  
  // NPU优化的矩阵乘法
  %C = boas.matmul %A, %B {npu_opt = #boas.npu_opt<128,256,256,true,"diagonal">}
      : (!boas.tensor<1024x1024xf32@npu>, !boas.tensor<1024x1024xf32@npu>) 
      -> !boas.tensor<1024x1024xf32@npu>
  
  return
}
```

### 优化Pipeline

```bash
# 运行Boas优化pipeline
mlir-opt input.mlir \
  --boas-matrix-opt \
  --boas-npu-opt \
  --boas-to-linalg \
  --canonicalize \
  --linalg-tile \
  --npu-kernel-gen \
  -o optimized.mlir
```

## 与现有架构集成

### MLIRGen集成

```cpp
// 扩展的MLIRGen支持Boas dialect
class BoasMLIRGen : public MLIRGen {
public:
  mlir::Value generateBoasMatmul(const MatmulExprAST* expr);
  mlir::Value generateBoasTensorCreate(const TensorCreateExprAST* expr);
  void runBoasOptimizations(ModuleOp module);
};
```

### NPU Backend集成

```cpp
// NPU backend自动选择Boas dialect路径
if (NPUBackend::isAvailable()) {
    // 使用Boas dialect生成高级操作
    result = generateBoasDialectMatmul(expr, lhs, rhs, m, n, k);
} else {
    // 回退到传统CPU路径
    result = createOptimizedMatmul(lhs, rhs, result, m, n, k);
}
```

## 优势对比

### vs. Triton依赖

| 特性 | Boas Dialect | Triton依赖 |
|------|-------------|------------|
| 控制程度 | 完全控制 | 依赖外部项目 |
| 语义匹配 | 直接对应Boas语义 | 需要转换 |
| 优化空间 | 可实现任意优化 | 受限于Triton |
| 设备感知 | 内置设备信息 | 需要额外处理 |
| 扩展性 | 易于扩展 | 受Triton限制 |
| 维护成本 | 自主维护 | 跟随Triton版本 |

### vs. 直接标准Dialect

| 特性 | Boas Dialect | 标准Dialect |
|------|-------------|-------------|
| 抽象层次 | 高级语义 | 低级操作 |
| 优化机会 | 语义级优化 | 语法级优化 |
| 代码可读性 | 高 | 低 |
| 调试难度 | 容易 | 困难 |
| 性能潜力 | 更好 | 有限 |

## 扩展指南

### 添加新操作

1. **定义操作**（在BoasOps.td中）：
```tablegen
def Boas_ConvOp : Boas_MatrixOp<"conv2d"> {
  let summary = "2D卷积操作";
  // ... 定义参数、结果、验证等
};
```

2. **实现操作**（在BoasOps.cpp中）：
```cpp
LogicalResult ConvOp::verify() {
  // 验证逻辑
}
```

3. **添加Lowering**（在相应Pass中）：
```cpp
struct ConvOpLowering : public OpConversionPattern<ConvOp> {
  LogicalResult matchAndRewrite(...) override {
    // lowering逻辑
  }
};
```

### 添加新的优化Pass

1. **定义Pass**（在BoasPasses.td中）
2. **实现Pass**（创建新的.cpp文件）
3. **注册Pass**（在BoasIntegration.cpp中）

## 调试和性能分析

### 查看生成的MLIR

```bash
# 查看Boas dialect IR
./boas-compiler --emit=boas-dialect input.bs

# 查看lowering后的IR
./boas-compiler --emit=linalg input.bs

# 查看最终优化的IR
./boas-compiler --emit=llvm input.bs
```

### 调试工具

```bash
# 启用详细日志
export BOAS_DEBUG=1

# 保存中间IR
./boas-compiler --save-temps input.bs

# 验证IR正确性
mlir-opt --verify-diagnostics generated.mlir
```

## 性能优化建议

### 1. 操作融合

Boas dialect支持高级操作融合：
```mlir
// 可融合的操作序列
%tmp = boas.matmul %A, %B
%result = boas.add %tmp, %C

// 融合后
%result = boas.fused_matmul_add %A, %B, %C
```

### 2. 内存优化

- 使用BF16减少内存带宽
- 实现in-place操作
- 优化数据布局

### 3. 并行化

- 自动分析并行化机会
- 生成多核优化代码
- 支持异步执行

## 总结

Boas MLIR Dialect提供了一个完整的、自主可控的编译器基础设施，专门为Boas语言和昇腾NPU优化设计。相比依赖外部Triton，它提供了：

1. **更好的控制**：完全自主的优化策略
2. **更高的性能**：语义级优化机会
3. **更强的扩展性**：易于添加新特性
4. **更好的集成**：与Boas语言完美匹配

这个dialect是Boas语言向前发展的重要基石，为实现高性能NPU计算提供了坚实的基础。
