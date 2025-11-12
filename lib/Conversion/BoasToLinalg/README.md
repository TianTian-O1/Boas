# Boas到Linalg Lowering Pass

## 概述

本目录包含将Boas Dialect operations降低到Linalg Dialect的转换pass。

## 文件结构

```
Conversion/BoasToLinalg/
├── BoasToLinalg.h              # Pass接口定义
├── BoasToLinalg.cpp            # Pass实现和转换逻辑
├── CMakeLists.txt              # 构建配置
└── README.md                   # 本文件
```

## 转换映射

### MatMul操作

**输入 (Boas)**:
```mlir
%C = boas.matmul %A, %B : !boas.tensor<2x3xf32>, !boas.tensor<3x4xf32>
                        -> !boas.tensor<2x4xf32>
```

**输出 (Linalg)**:
```mlir
%empty = tensor.empty() : tensor<2x4xf32>
%zero = arith.constant 0.0 : f32
%init = linalg.fill ins(%zero : f32) outs(%empty : tensor<2x4xf32>) -> tensor<2x4xf32>
%C = linalg.matmul ins(%A, %B : tensor<2x3xf32>, tensor<3x4xf32>)
                    outs(%init : tensor<2x4xf32>) -> tensor<2x4xf32>
```

**关键点**:
1. Linalg matmul需要预初始化的输出tensor
2. 输出必须初始化为0（因为matmul是累加操作）
3. 类型从`!boas.tensor`转换为标准`tensor`

## 使用方法

### 1. 使用boas-opt

```bash
# 运行lowering pass
boas-opt --convert-boas-to-linalg input.mlir -o output.mlir

# 查看pass帮助
boas-opt --help | grep "convert-boas-to-linalg"
```

### 2. 在Pass Pipeline中使用

```bash
# 完整pipeline
boas-opt input.mlir \
  --convert-boas-to-linalg \
  --linalg-fuse-elementwise-ops \
  --linalg-bufferize \
  -o output.mlir
```

### 3. 编程方式使用

```cpp
#include "Boas/Conversion/BoasToLinalg/BoasToLinalg.h"

void runConversion(ModuleOp module) {
  PassManager pm(module.getContext());
  pm.addPass(mlir::boas::createConvertBoasToLinalgPass());

  if (failed(pm.run(module))) {
    llvm::errs() << "Conversion failed\n";
  }
}
```

## 测试

### 运行测试

```bash
cd build
ninja check-boas-conversion

# 或单独运行
lit test/Conversion/boas-to-linalg-matmul.mlir
```

### Standalone测试程序

编译并运行standalone conversion test：

```bash
cd build
ninja standalone-matmul-conversion
./tools/standalone-conversion-test/standalone-matmul-conversion
```

这将生成一个示例函数，展示MatMul的完整转换。

## 转换逻辑详解

### 核心转换函数

文件: `BoasToLinalg.cpp`

```cpp
Value convertMatMulOp(OpBuilder &builder, Location loc,
                       Value lhs, Value rhs,
                       RankedTensorType resultType) {
  // 1. 创建空tensor
  Value emptyTensor = builder.create<tensor::EmptyOp>(...);

  // 2. 创建零常量
  Value zero = builder.create<arith::ConstantOp>(...);

  // 3. 初始化为零
  Value initTensor = builder.create<linalg::FillOp>(...);

  // 4. 执行matmul
  Value result = builder.create<linalg::MatmulOp>(...);

  return result;
}
```

### 类型转换器

```cpp
class BoasTypeConverter : public TypeConverter {
public:
  BoasTypeConverter() {
    // Boas TensorType -> RankedTensorType
    addConversion([](boas::TensorType type) -> Type {
      return RankedTensorType::get(type.getShape(),
                                    type.getElementType());
    });
  }
};
```

## 性能考虑

### IR大小
每个MatMul操作转换为4个Linalg/Tensor operations：
- 1x tensor.empty
- 1x arith.constant
- 1x linalg.fill
- 1x linalg.matmul

### 运行时开销
- Zero初始化: O(M*N)
- MatMul计算: O(M*K*N)
- 总开销: O(M*K*N) (dominated by matmul)

### 优化机会
后续Pass可以优化掉不必要的初始化：
```
linalg-fuse-fill-into-matmul  # 融合fill和matmul
linalg-eliminate-dead-allocs  # 消除死代码
```

## 扩展指南

### 添加新Operation转换

1. 在`BoasToLinalg.cpp`中添加conversion pattern:

```cpp
struct NewOpLowering : public OpConversionPattern<boas::NewOp> {
  LogicalResult matchAndRewrite(...) const override {
    // 实现转换逻辑
    return success();
  }
};
```

2. 在`populateBoasToLinalgConversionPatterns`中注册pattern:

```cpp
void populateBoasToLinalgConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<MatMulOpLowering, NewOpLowering>(
      patterns.getContext());
}
```

### 支持新类型

在`BoasTypeConverter`中添加转换规则：

```cpp
addConversion([](boas::NewType type) -> Type {
  // 转换逻辑
  return convertedType;
});
```

## 调试

### 打印转换前后的IR

```bash
boas-opt --convert-boas-to-linalg \
  --mlir-print-ir-before-all \
  --mlir-print-ir-after-all \
  input.mlir
```

### 验证生成的IR

```bash
boas-opt --convert-boas-to-linalg --verify-each input.mlir
```

### 查看Pass统计

```bash
boas-opt --convert-boas-to-linalg \
  --mlir-pass-statistics \
  input.mlir
```

## 已知限制

当前版本限制：
- ✅ 支持2D矩阵乘法
- ⚠️ 不支持batch matmul
- ⚠️ 不支持transpose选项
- ⚠️ 固定zero初始化

计划功能：
- 🔜 Batch MatMul (3D+ tensors)
- 🔜 Transpose flags (A^T, B^T)
- 🔜 Alpha/Beta scaling (GEMM)
- 🔜 优化初始化策略

## 相关文档

- [完整设计文档](../../../docs/BoasToLinalgDesign.md)
- [Boas Dialect定义](../../Dialect/Boas/IR/)
- [测试用例](../../../test/Conversion/)

## 依赖

- MLIR Linalg Dialect
- MLIR Tensor Dialect
- MLIR Arith Dialect
- MLIR Func Dialect

## 维护者

- Boas-NPU Team
- 最后更新: 2025-11-12
