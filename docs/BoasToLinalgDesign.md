# Boas到Linalg Lowering Pass设计文档

## 概述

本文档描述Boas Dialect到Linalg Dialect的Lowering Pass设计和实现。

## 1. Pass架构

### 1.1 Pass层次结构

```
Boas-NPU Compilation Pipeline:

┌─────────────────────┐
│   Boas Dialect      │  <- 高层抽象 (Python风格)
│   (MatMulOp, etc)   │
└──────────┬──────────┘
           │ BoasToLinalgPass
           ↓
┌─────────────────────┐
│  Linalg Dialect     │  <- 结构化操作
│  (linalg.matmul)    │
└──────────┬──────────┘
           │ LinalgToHFusionPass (BiShengIR)
           ↓
┌─────────────────────┐
│  HFusion Dialect    │  <- 算子融合
└──────────┬──────────┘
           │ HFusionToHIVMPass
           ↓
┌─────────────────────┐
│   HIVM Dialect      │  <- 虚拟机IR
└──────────┬──────────┘
           │ HIVMToTritonPass
           ↓
┌─────────────────────┐
│  Triton/LIR         │  <- NPU后端
│  (昇腾)             │
└─────────────────────┘
```

### 1.2 BoasToLinalg Pass职责

**输入**: Boas Dialect operations
**输出**: Linalg Dialect operations
**任务**:
1. 类型转换: `!boas.tensor<MxN>` → `tensor<MxN>`
2. Operation转换: `boas.matmul` → `linalg.matmul`
3. 辅助操作生成: tensor.empty, linalg.fill

## 2. MatMul Operation转换

### 2.1 转换规则

#### Boas MatMul语义
```mlir
%C = boas.matmul %A, %B : !boas.tensor<MxK>, !boas.tensor<KxN>
                        -> !boas.tensor<MxN>
```

**语义**: C[i,j] = Σ(k=0 to K-1) A[i,k] * B[k,j]

#### Linalg MatMul语义
```mlir
%C = linalg.matmul ins(%A, %B : tensor<MxK>, tensor<KxN>)
                    outs(%C_init : tensor<MxN>) -> tensor<MxN>
```

**语义**: C_out[i,j] = C_init[i,j] + Σ(k) A[i,k] * B[k,j]

**关键差异**: Linalg的matmul是累加操作，需要初始化输出！

### 2.2 转换步骤

完整转换包含4个步骤：

```mlir
// Step 1: 创建空tensor (shape = [M, N])
%empty = tensor.empty() : tensor<MxN>

// Step 2: 创建零常量
%zero = arith.constant 0.0 : element_type

// Step 3: 初始化输出为零
%init = linalg.fill ins(%zero : element_type)
                     outs(%empty : tensor<MxN>) -> tensor<MxN>

// Step 4: 执行矩阵乘法
%result = linalg.matmul ins(%A, %B : tensor<MxK>, tensor<KxN>)
                         outs(%init : tensor<MxN>) -> tensor<MxN>
```

### 2.3 代码实现

**文件**: `lib/Conversion/BoasToLinalg/BoasToLinalg.cpp`

```cpp
Value convertMatMulOp(OpBuilder &builder, Location loc,
                       Value lhs, Value rhs, RankedTensorType resultType) {
  // Step 1: Create empty tensor
  Value emptyTensor = builder.create<tensor::EmptyOp>(
      loc, resultType.getShape(), resultType.getElementType());

  // Step 2: Create zero constant
  Value zero;
  if (resultType.getElementType().isF32()) {
    zero = builder.create<arith::ConstantOp>(
        loc, builder.getF32FloatAttr(0.0));
  } else if (resultType.getElementType().isF64()) {
    zero = builder.create<arith::ConstantOp>(
        loc, builder.getF64FloatAttr(0.0));
  }

  // Step 3: Fill with zeros
  Value initTensor = builder.create<linalg::FillOp>(
      loc, ValueRange{zero}, ValueRange{emptyTensor}).getResult(0);

  // Step 4: MatMul
  Value result = builder.create<linalg::MatmulOp>(
      loc, resultType, ValueRange{lhs, rhs}, ValueRange{initTensor})
      .getResult(0);

  return result;
}
```

## 3. 类型转换

### 3.1 Tensor类型映射

| Boas Type | MLIR Type | 说明 |
|-----------|-----------|------|
| `!boas.tensor<2x3xf32>` | `tensor<2x3xf32>` | 静态shape |
| `!boas.tensor<?x3xf32>` | `tensor<?x3xf32>` | 动态shape |
| `!boas.tensor<2x?xf32>` | `tensor<2x?xf32>` | 部分动态 |

### 3.2 元素类型支持

- ✅ `f32` - 单精度浮点
- ✅ `f64` - 双精度浮点
- 🚧 `i32` - 32位整数 (planned)
- 🚧 `bf16` - BFloat16 (planned)

### 3.3 类型转换器实现

```cpp
class BoasTypeConverter : public TypeConverter {
public:
  BoasTypeConverter() {
    // Boas TensorType -> RankedTensorType
    addConversion([](boas::TensorType type) -> Type {
      return RankedTensorType::get(
          type.getShape(),
          type.getElementType());
    });

    // Keep other types unchanged
    addConversion([](Type type) { return type; });
  }
};
```

## 4. Pattern Matching

### 4.1 Conversion Pattern框架

```cpp
struct MatMulOpLowering : public OpConversionPattern<boas::MatMulOp> {
  using OpConversionPattern<boas::MatMulOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      boas::MatMulOp op,
      OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    Location loc = op.getLoc();
    auto resultType = typeConverter->convertType(op.getType())
                          .cast<RankedTensorType>();

    // Use helper function
    Value result = convertMatMulOp(
        rewriter, loc,
        adaptor.getLhs(), adaptor.getRhs(),
        resultType);

    rewriter.replaceOp(op, result);
    return success();
  }
};
```

### 4.2 Pattern应用流程

```
1. Pattern Matching
   ↓
2. matchAndRewrite()
   ↓
3. Type Conversion (Boas → Linalg)
   ↓
4. Operation Generation (helper function)
   ↓
5. Replace Original Op
```

## 5. Pass执行

### 5.1 Pass注册

**文件**: `tools/boas-opt/boas-opt.cpp`

```cpp
void registerBoasConversionPasses() {
  PassRegistration<ConvertBoasToLinalgPass>();
}
```

### 5.2 Pass使用

```bash
# 运行Lowering pass
boas-opt --convert-boas-to-linalg input.mlir -o output.mlir

# 查看Pass帮助
boas-opt --help | grep boas
```

### 5.3 Pass Pipeline

```bash
# 完整pipeline示例
boas-opt input.mlir \
  --convert-boas-to-linalg \
  --linalg-fuse-elementwise-ops \
  --linalg-bufferize \
  -o output.mlir
```

## 6. 测试策略

### 6.1 测试用例结构

**文件**: `test/Conversion/boas-to-linalg-matmul.mlir`

```mlir
// Test 1: Basic 2D matmul
func.func @matmul_2x3_3x4(...) -> ... {
  // CHECK: tensor.empty
  // CHECK: arith.constant 0.0
  // CHECK: linalg.fill
  // CHECK: linalg.matmul
  ...
}

// Test 2: Square matrices
// Test 3: Large dimensions
// Test 4: f64 element type
```

### 6.2 验证点

每个测试验证：
1. ✅ 输出shape正确
2. ✅ 元素类型保持
3. ✅ 生成所有必要操作
4. ✅ 操作顺序正确

### 6.3 FileCheck模式

```mlir
// CHECK-LABEL: func @test_name
// CHECK: %[[VAR:.*]] = operation
// CHECK-SAME: attributes
// CHECK-NEXT: another operation
```

## 7. 优化机会

### 7.1 当前实现

基础转换，无优化：

```mlir
%empty → %zero → %fill → %matmul
```

### 7.2 优化方向

#### 7.2.1 Zero初始化消除
如果检测到输出tensor已经是零，跳过fill：

```cpp
if (!isAlreadyZero(initTensor)) {
  initTensor = builder.create<linalg::FillOp>(...);
}
```

#### 7.2.2 In-place操作
如果输出tensor可以原地修改：

```mlir
%result = linalg.matmul ins(%A, %B)
                         outs(%existing_tensor)
                         -> tensor<MxN>
```

#### 7.2.3 Tile大小优化
根据NPU特性选择最优tile size：

```cpp
// 昇腾NPU偏好32的倍数
const int64_t TILE_SIZE = 32;
```

## 8. 错误处理

### 8.1 Shape不匹配检测

```cpp
// 在转换前验证
if (lhsShape[1] != rhsShape[0]) {
  return op.emitError("incompatible matmul dimensions");
}
```

### 8.2 类型不支持

```cpp
if (!isSupportedElementType(elementType)) {
  return op.emitError("unsupported element type: ") << elementType;
}
```

## 9. 性能考虑

### 9.1 IR生成开销

- 每个MatMul生成4个新operation
- 需要创建2个常量 (shape indices, zero)
- SSA value管理开销

### 9.2 后续优化Pass

Lowering之后的优化pipeline：
```
BoasToLinalg
  ↓
LinalgFusion         (融合相邻操作)
  ↓
LinalgTiling         (分块优化)
  ↓
LinalgToLoops        (降低到循环)
  ↓
LoopOptimization     (循环优化)
```

## 10. 扩展性

### 10.1 添加新Operation转换

模板代码：

```cpp
struct NewOpLowering : public OpConversionPattern<boas::NewOp> {
  using OpConversionPattern<boas::NewOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(...) const override {
    // 1. Get operands
    // 2. Convert types
    // 3. Generate Linalg ops
    // 4. Replace original op
    return success();
  }
};
```

### 10.2 Dialect依赖管理

在Pass中声明依赖：

```cpp
void getDependentDialects(DialectRegistry &registry) const override {
  registry.insert<
    linalg::LinalgDialect,
    tensor::TensorDialect,
    arith::ArithDialect,
    // 添加新dialect
    your::NewDialect
  >();
}
```

## 11. 调试技巧

### 11.1 打印IR

```cpp
// 在转换前后打印
op.dump();  // 转换前
result.getDefiningOp()->dump();  // 转换后
```

### 11.2 Pass Manager诊断

```bash
# 启用详细输出
boas-opt --convert-boas-to-linalg --mlir-print-ir-before-all input.mlir

# 只打印特定pass
boas-opt --mlir-print-ir-after=convert-boas-to-linalg input.mlir
```

### 11.3 验证IR正确性

```cpp
// 在Pass中验证
if (failed(verify(module))) {
  return signalPassFailure();
}
```

## 12. 已知限制

### 12.1 当前版本限制

- ⚠️ 仅支持2D矩阵乘法
- ⚠️ 不支持batch matmul
- ⚠️ 不支持transpose标志
- ⚠️ 固定zero初始化（不支持自定义初值）

### 12.2 计划增强

- 🔜 Batch MatMul支持
- 🔜 MatMul + Bias融合
- 🔜 Transpose选项
- 🔜 Alpha/Beta系数 (GEMM)

## 13. 与BiShengIR集成

### 13.1 下游Pass

BoasToLinalg之后，IR进入BiShengIR pipeline：

```
Linalg Dialect
  ↓ LinalgToHFusionPass
HFusion Dialect  (BiShengIR)
  ↓ HFusionToHIVMPass
HIVM Dialect     (BiShengIR)
  ↓ HIVMToTritonPass
Triton/LIR       (昇腾后端)
```

### 13.2 接口契约

BoasToLinalg必须保证：
- ✅ 生成合法的Linalg IR
- ✅ 保持shape和type信息
- ✅ 可被后续pass处理

## 14. 总结

### 14.1 核心价值

1. **抽象分离**: 用户使用高层Boas语法，编译器处理底层细节
2. **可优化**: Linalg提供丰富的优化机会
3. **可移植**: 标准MLIR Dialect，易于集成
4. **可扩展**: 清晰的pattern框架

### 14.2 代码位置

| 组件 | 文件 |
|------|------|
| Pass头文件 | `include/Boas/Conversion/BoasToLinalg/BoasToLinalg.h` |
| Pass实现 | `lib/Conversion/BoasToLinalg/BoasToLinalg.cpp` |
| Pass定义 | `include/Boas/Conversion/Passes.td` |
| 测试用例 | `test/Conversion/boas-to-linalg-matmul.mlir` |
| 构建配置 | `lib/Conversion/BoasToLinalg/CMakeLists.txt` |

### 14.3 下一步

- [ ] 完成Boas Dialect编译问题修复
- [ ] 实现完整的pattern matching
- [ ] 添加更多operation转换
- [ ] 集成到完整pipeline
- [ ] 性能benchmarking

---

**文档版本**: 1.0
**最后更新**: 2025-11-12
**维护者**: Boas-NPU Team
