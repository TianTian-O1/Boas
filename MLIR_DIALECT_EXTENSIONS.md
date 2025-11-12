# Boas MLIR Dialect Extensions Plan

**Version**: 0.2.0
**Status**: Design Phase
**Last Updated**: 2025-11-13

---

## 🎯 Overview

This document outlines the plan to extend the current Boas MLIR dialect from supporting only matrix multiplication to a full-featured programming language dialect.

**Current State**: Basic MatMul operation (v0.1.0)
**Target State**: Complete language dialect with control flow, memory management, and hardware acceleration (v0.2.0+)

---

## 📊 Current Boas Dialect (v0.1.0)

### Existing Operations

```mlir
// include/Boas/Dialect/Boas/IR/BoasOps.td
def Boas_MatMulOp : Boas_Op<"matmul", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Matrix multiplication operation";
  let arguments = (ins Boas_TensorType:$lhs, Boas_TensorType:$rhs);
  let results = (outs Boas_TensorType:$result);
  let hasVerifier = 1;
}
```

### Existing Types

```mlir
// include/Boas/Dialect/Boas/IR/BoasTypes.td
def Boas_TensorType : Boas_Type<"Tensor"> {
  let summary = "Boas tensor type";
  let mnemonic = "tensor";
}
```

---

## 🚀 Planned Extensions

### Phase 1: Basic Operations (Priority: HIGH)

#### 1.1 Arithmetic Operations

```tablegen
// BoasOps.td additions

def Boas_AddOp : Boas_Op<"add", [Pure, Commutative]> {
  let summary = "Addition operation";
  let arguments = (ins Boas_Type:$lhs, Boas_Type:$rhs);
  let results = (outs Boas_Type:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def Boas_SubOp : Boas_Op<"sub", [Pure]> {
  let summary = "Subtraction operation";
  let arguments = (ins Boas_Type:$lhs, Boas_Type:$rhs);
  let results = (outs Boas_Type:$result);
}

def Boas_MulOp : Boas_Op<"mul", [Pure, Commutative]> {
  let summary = "Element-wise multiplication";
  let arguments = (ins Boas_Type:$lhs, Boas_Type:$rhs);
  let results = (outs Boas_Type:$result);
}

def Boas_DivOp : Boas_Op<"div", [Pure]> {
  let summary = "Division operation";
  let arguments = (ins Boas_Type:$lhs, Boas_Type:$rhs);
  let results = (outs Boas_Type:$result);
}
```

**Example MLIR**:
```mlir
func.func @arithmetic_demo(%a: !boas.tensor<f32>, %b: !boas.tensor<f32>) -> !boas.tensor<f32> {
  %0 = boas.add %a, %b : !boas.tensor<f32>
  %1 = boas.mul %0, %b : !boas.tensor<f32>
  return %1 : !boas.tensor<f32>
}
```

#### 1.2 Comparison Operations

```tablegen
def Boas_CmpOp : Boas_Op<"cmp", [Pure]> {
  let summary = "Comparison operation";
  let arguments = (ins
    Boas_Type:$lhs,
    Boas_Type:$rhs,
    Boas_CmpPredicateAttr:$predicate
  );
  let results = (outs Boas_BoolType:$result);
}

def Boas_CmpPredicateAttr : I32EnumAttr<"CmpPredicate",
    "Comparison predicate", [
      I32EnumAttrCase<"EQ", 0, "eq">,   // Equal
      I32EnumAttrCase<"NE", 1, "ne">,   // Not equal
      I32EnumAttrCase<"LT", 2, "lt">,   // Less than
      I32EnumAttrCase<"LE", 3, "le">,   // Less or equal
      I32EnumAttrCase<"GT", 4, "gt">,   // Greater than
      I32EnumAttrCase<"GE", 5, "ge">,   // Greater or equal
    ]> {
  let cppNamespace = "::mlir::boas";
}
```

**Example MLIR**:
```mlir
%cond = boas.cmp lt, %a, %b : !boas.tensor<f32> -> !boas.bool
```

### Phase 2: Control Flow (Priority: HIGH)

#### 2.1 Conditional Operations

```tablegen
def Boas_IfOp : Boas_Op<"if", [
    RecursiveMemoryEffects,
    SingleBlockImplicitTerminator<"boas::YieldOp">
  ]> {
  let summary = "Conditional execution";
  let arguments = (ins Boas_BoolType:$condition);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$thenRegion, AnyRegion:$elseRegion);

  let assemblyFormat = [{
    $condition `then` $thenRegion (`else` $elseRegion^)?
    attr-dict `:` functional-type($condition, $results)
  }];
}

def Boas_YieldOp : Boas_Op<"yield", [Pure, Terminator]> {
  let summary = "Yield values from a region";
  let arguments = (ins Variadic<AnyType>:$values);
  let assemblyFormat = "attr-dict ($values^ `:` type($values))?";
}
```

**Example MLIR**:
```mlir
%result = boas.if %condition then {
  %true_val = boas.compute_true()
  boas.yield %true_val : !boas.tensor<f32>
} else {
  %false_val = boas.compute_false()
  boas.yield %false_val : !boas.tensor<f32>
} : !boas.bool -> !boas.tensor<f32>
```

#### 2.2 Loop Operations

```tablegen
def Boas_ForOp : Boas_Op<"for", [
    RecursiveMemoryEffects,
    SingleBlockImplicitTerminator<"boas::YieldOp">
  ]> {
  let summary = "For loop operation";
  let arguments = (ins
    Index:$lowerBound,
    Index:$upperBound,
    Index:$step,
    Variadic<AnyType>:$initArgs
  );
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$bodyRegion);

  let builders = [
    OpBuilder<(ins "Value":$lb, "Value":$ub, "Value":$step)>
  ];
}

def Boas_WhileOp : Boas_Op<"while", [RecursiveMemoryEffects]> {
  let summary = "While loop operation";
  let arguments = (ins Variadic<AnyType>:$initArgs);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$condition, SizedRegion<1>:$body);
}
```

**Example MLIR**:
```mlir
// For loop: sum = 0; for i in 0..10: sum += i
%sum = boas.for %i = %c0 to %c10 step %c1 iter_args(%arg = %c0_i32) -> i32 {
  %next = boas.add %arg, %i : i32
  boas.yield %next : i32
}

// While loop
%result = boas.while (%arg = %init) : (i32) -> i32 {
  // Condition
  %cond = boas.cmp lt, %arg, %limit : i32 -> i1
  boas.condition %cond
} do {
  ^bb0(%arg: i32):
  %next = boas.add %arg, %c1 : i32
  boas.yield %next : i32
}
```

### Phase 3: Memory Operations (Priority: HIGH)

#### 3.1 Memory Allocation and Access

```tablegen
def Boas_AllocOp : Boas_Op<"alloc", [MemoryEffects<[MemAlloc]>]> {
  let summary = "Allocate memory";
  let arguments = (ins Variadic<Index>:$sizes);
  let results = (outs Boas_PtrType:$result);
}

def Boas_DeallocOp : Boas_Op<"dealloc", [MemoryEffects<[MemFree]>]> {
  let summary = "Deallocate memory";
  let arguments = (ins Boas_PtrType:$ptr);
}

def Boas_LoadOp : Boas_Op<"load", [MemoryEffects<[MemRead]>]> {
  let summary = "Load value from memory";
  let arguments = (ins Boas_PtrType:$ptr, Variadic<Index>:$indices);
  let results = (outs AnyType:$result);
}

def Boas_StoreOp : Boas_Op<"store", [MemoryEffects<[MemWrite]>]> {
  let summary = "Store value to memory";
  let arguments = (ins AnyType:$value, Boas_PtrType:$ptr, Variadic<Index>:$indices);
}
```

**Example MLIR**:
```mlir
// Allocate array of 100 f32 values
%ptr = boas.alloc(%c100) : !boas.ptr<f32>

// Store value at index 10
boas.store %value, %ptr[%c10] : f32, !boas.ptr<f32>

// Load value from index 10
%loaded = boas.load %ptr[%c10] : !boas.ptr<f32> -> f32

// Deallocate
boas.dealloc %ptr : !boas.ptr<f32>
```

### Phase 4: Function Operations (Priority: MEDIUM)

#### 4.1 Function Definition and Calls

```tablegen
def Boas_FuncOp : Boas_Op<"func", [
    IsolatedFromAbove,
    FunctionOpInterface,
    Symbol
  ]> {
  let summary = "Boas function definition";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type
  );
  let regions = (region AnyRegion:$body);

  let builders = [
    OpBuilder<(ins "StringRef":$name, "FunctionType":$type)>
  ];
}

def Boas_CallOp : Boas_Op<"call", []> {
  let summary = "Call a function";
  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    Variadic<AnyType>:$operands
  );
  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, $results)
  }];
}

def Boas_ReturnOp : Boas_Op<"return", [Pure, Terminator]> {
  let summary = "Return from function";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}
```

**Example MLIR**:
```mlir
boas.func @add(%a: i32, %b: i32) -> i32 {
  %result = boas.add %a, %b : i32
  boas.return %result : i32
}

func.func @main() {
  %a = arith.constant 10 : i32
  %b = arith.constant 20 : i32
  %sum = boas.call @add(%a, %b) : (i32, i32) -> i32
  return
}
```

### Phase 5: Tensor/Neural Network Operations (Priority: MEDIUM)

#### 5.1 Neural Network Ops

```tablegen
def Boas_Conv2DOp : Boas_Op<"conv2d", [Pure]> {
  let summary = "2D convolution";
  let arguments = (ins
    Boas_TensorType:$input,   // [N, C_in, H, W]
    Boas_TensorType:$kernel,  // [C_out, C_in, KH, KW]
    OptionalAttr<I64ArrayAttr>:$stride,
    OptionalAttr<I64ArrayAttr>:$padding
  );
  let results = (outs Boas_TensorType:$output);
}

def Boas_ReLUOp : Boas_Op<"relu", [Pure]> {
  let summary = "ReLU activation function";
  let arguments = (ins Boas_TensorType:$input);
  let results = (outs Boas_TensorType:$output);
}

def Boas_SoftmaxOp : Boas_Op<"softmax", [Pure]> {
  let summary = "Softmax activation function";
  let arguments = (ins Boas_TensorType:$input, I64Attr:$dim);
  let results = (outs Boas_TensorType:$output);
}

def Boas_Pool2DOp : Boas_Op<"pool2d", [Pure]> {
  let summary = "2D pooling (max or average)";
  let arguments = (ins
    Boas_TensorType:$input,
    I64ArrayAttr:$kernel_size,
    Boas_PoolTypeAttr:$pool_type
  );
  let results = (outs Boas_TensorType:$output);
}
```

**Example MLIR**:
```mlir
// CNN forward pass
func.func @cnn_forward(%input: !boas.tensor<1x3x224x224xf32>) -> !boas.tensor<1x10xf32> {
  // Conv layer
  %conv1 = boas.conv2d %input, %kernel1 {stride = [1, 1], padding = [1, 1]}
    : !boas.tensor<1x3x224x224xf32> -> !boas.tensor<1x64x224x224xf32>

  // ReLU
  %relu1 = boas.relu %conv1 : !boas.tensor<1x64x224x224xf32>

  // Max pooling
  %pool1 = boas.pool2d %relu1 {kernel_size = [2, 2], type = "max"}
    : !boas.tensor<1x64x224x224xf32> -> !boas.tensor<1x64x112x112xf32>

  // ... more layers ...

  // Final softmax
  %output = boas.softmax %fc {dim = 1} : !boas.tensor<1x10xf32>
  return %output : !boas.tensor<1x10xf32>
}
```

### Phase 6: Device/Hardware Operations (Priority: HIGH)

#### 6.1 Device Management

```tablegen
def Boas_GetDeviceOp : Boas_Op<"get_device", []> {
  let summary = "Get device handle";
  let arguments = (ins StrAttr:$device_type, I32Attr:$device_id);
  let results = (outs Boas_DeviceType:$device);
}

def Boas_ToDeviceOp : Boas_Op<"to_device", [MemoryEffects<[MemWrite]>]> {
  let summary = "Transfer tensor to device";
  let arguments = (ins Boas_TensorType:$input, Boas_DeviceType:$device);
  let results = (outs Boas_TensorType:$output);
}

def Boas_ExecuteOnOp : Boas_Op<"execute_on", [
    RecursiveMemoryEffects,
    SingleBlockImplicitTerminator<"boas::YieldOp">
  ]> {
  let summary = "Execute region on specific device";
  let arguments = (ins Boas_DeviceType:$device);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);
}
```

**Example MLIR**:
```mlir
// Get NPU device
%npu = boas.get_device "npu", 0 : !boas.device

// Transfer to NPU
%tensor_npu = boas.to_device %tensor, %npu
  : !boas.tensor<100x100xf32> -> !boas.tensor<100x100xf32>

// Execute on NPU
%result = boas.execute_on %npu {
  %r = boas.matmul %a_npu, %b_npu : !boas.tensor<100x100xf32>
  boas.yield %r : !boas.tensor<100x100xf32>
} : !boas.device -> !boas.tensor<100x100xf32>
```

### Phase 7: Async/Concurrency Operations (Priority: MEDIUM)

#### 7.1 Asynchronous Execution

```tablegen
def Boas_AsyncOp : Boas_Op<"async", [
    RecursiveMemoryEffects,
    SingleBlockImplicitTerminator<"boas::YieldOp">
  ]> {
  let summary = "Asynchronous execution";
  let results = (outs Boas_FutureType:$future);
  let regions = (region SizedRegion<1>:$body);
}

def Boas_AwaitOp : Boas_Op<"await", []> {
  let summary = "Wait for async result";
  let arguments = (ins Boas_FutureType:$future);
  let results = (outs AnyType:$result);
}

def Boas_ChannelCreateOp : Boas_Op<"channel_create", []> {
  let summary = "Create a channel";
  let arguments = (ins I32Attr:$buffer_size);
  let results = (outs Boas_ChannelType:$channel);
}

def Boas_ChannelSendOp : Boas_Op<"channel_send", []> {
  let summary = "Send value to channel";
  let arguments = (ins AnyType:$value, Boas_ChannelType:$channel);
}

def Boas_ChannelReceiveOp : Boas_Op<"channel_receive", []> {
  let summary = "Receive value from channel";
  let arguments = (ins Boas_ChannelType:$channel);
  let results = (outs AnyType:$value);
}
```

**Example MLIR**:
```mlir
// Async execution
%future = boas.async {
  %result = boas.heavy_compute %input
  boas.yield %result : !boas.tensor<f32>
} : !boas.future<!boas.tensor<f32>>

// Wait for result
%value = boas.await %future : !boas.future<!boas.tensor<f32>> -> !boas.tensor<f32>

// Channel communication
%ch = boas.channel_create {buffer_size = 10} : !boas.channel<i32>
boas.channel_send %value, %ch : i32, !boas.channel<i32>
%received = boas.channel_receive %ch : !boas.channel<i32> -> i32
```

---

## 🏗️ New Type System

### Extended Types Definition

```tablegen
// BoasTypes.td extensions

def Boas_BoolType : Boas_Type<"Bool"> {
  let summary = "Boolean type";
  let mnemonic = "bool";
}

def Boas_IntType : Boas_Type<"Int"> {
  let summary = "Integer type";
  let mnemonic = "int";
  let parameters = (ins "unsigned":$width, "bool":$isSigned);
}

def Boas_FloatType : Boas_Type<"Float"> {
  let summary = "Floating point type";
  let mnemonic = "float";
  let parameters = (ins "unsigned":$width);
}

def Boas_PtrType : Boas_Type<"Ptr"> {
  let summary = "Pointer type";
  let mnemonic = "ptr";
  let parameters = (ins "Type":$elementType);
}

def Boas_DeviceType : Boas_Type<"Device"> {
  let summary = "Device handle type";
  let mnemonic = "device";
}

def Boas_FutureType : Boas_Type<"Future"> {
  let summary = "Future/promise type for async";
  let mnemonic = "future";
  let parameters = (ins "Type":$resultType);
}

def Boas_ChannelType : Boas_Type<"Channel"> {
  let summary = "Channel type for communication";
  let mnemonic = "channel";
  let parameters = (ins "Type":$elementType);
}

def Boas_FunctionType : Boas_Type<"Function"> {
  let summary = "Function type";
  let mnemonic = "func";
  let parameters = (ins
    ArrayRefParameter<"Type">:$inputs,
    ArrayRefParameter<"Type">:$outputs
  );
}
```

---

## 🔄 Lowering Passes

### New Conversion Passes Needed

#### 1. BoasToStandard Pass
```cpp
// lib/Conversion/BoasToStandard/BoasToStandard.cpp

class BoasAddOpLowering : public OpConversionPattern<boas::AddOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      boas::AddOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // Convert boas.add to arith.addf or arith.addi
    rewriter.replaceOpWithNewOp<arith::AddFOp>(op, adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};
```

#### 2. BoasControlFlowLowering Pass
```cpp
// Convert boas.if to scf.if
class BoasIfOpLowering : public OpConversionPattern<boas::IfOp> {
public:
  LogicalResult matchAndRewrite(
      boas::IfOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // Convert to scf.if
    auto scfIf = rewriter.create<scf::IfOp>(
        op.getLoc(), op.getResultTypes(), adaptor.getCondition(), true);

    // Clone then region
    rewriter.inlineRegionBefore(op.getThenRegion(), scfIf.getThenRegion(),
                                scfIf.getThenRegion().end());

    // Clone else region if exists
    if (!op.getElseRegion().empty()) {
      rewriter.inlineRegionBefore(op.getElseRegion(), scfIf.getElseRegion(),
                                  scfIf.getElseRegion().end());
    }

    rewriter.replaceOp(op, scfIf.getResults());
    return success();
  }
};
```

#### 3. BoasDeviceLowering Pass
```cpp
// Lower device operations to hardware-specific ops
class BoasToDeviceOpLowering : public OpConversionPattern<boas::ToDeviceOp> {
public:
  LogicalResult matchAndRewrite(
      boas::ToDeviceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // Determine device type and lower accordingly
    auto deviceAttr = op.getDevice();

    if (isNPUDevice(deviceAttr)) {
      // Lower to HIVM ops
      rewriter.replaceOpWithNewOp<hivm::MemcpyOp>(
          op, adaptor.getInput(), hivm::DeviceType::NPU);
    } else if (isGPUDevice(deviceAttr)) {
      // Lower to GPU ops
      rewriter.replaceOpWithNewOp<gpu::MemcpyOp>(
          op, adaptor.getInput(), gpu::AddressSpace::Global);
    }

    return success();
  }
};
```

---

## 📁 File Structure

### New Files to Create

```
include/Boas/Dialect/Boas/IR/
├── BoasOps.td                  [EXTEND] - Add new operations
├── BoasTypes.td                [EXTEND] - Add new types
├── BoasDialect.td              [EXTEND] - Dialect attributes
├── BoasAttributes.td           [NEW] - Custom attributes
└── BoasInterfaces.td           [NEW] - Operation interfaces

lib/Dialect/Boas/IR/
├── BoasOps.cpp                 [EXTEND] - Operation implementations
├── BoasTypes.cpp               [EXTEND] - Type implementations
├── BoasAttributes.cpp          [NEW] - Attribute implementations
└── BoasDialect.cpp             [EXTEND] - Dialect initialization

lib/Conversion/
├── BoasToStandard/             [NEW]
│   └── BoasToStandard.cpp
├── BoasToSCF/                  [NEW] - Control flow lowering
│   └── BoasToSCF.cpp
├── BoasToFunc/                 [NEW] - Function lowering
│   └── BoasToFunc.cpp
├── BoasToGPU/                  [NEW] - GPU lowering
│   └── BoasToGPU.cpp
└── BoasToLinalg/               [EXTEND] - Add more ops
    └── BoasToLinalg.cpp

lib/Transforms/                 [NEW]
├── BoasSimplify.cpp            - Simplification passes
├── BoasInlining.cpp            - Function inlining
├── BoasDeviceOpt.cpp           - Device-specific optimizations
└── BoasMemoryOpt.cpp           - Memory optimizations
```

---

## 🧪 Testing Strategy

### Test Files Structure

```
test/Dialect/Boas/
├── ops/
│   ├── arithmetic.mlir         [NEW] - Test arithmetic ops
│   ├── control_flow.mlir       [NEW] - Test if/for/while
│   ├── memory.mlir             [NEW] - Test alloc/load/store
│   ├── tensor.mlir             [NEW] - Test tensor ops
│   └── device.mlir             [NEW] - Test device ops
├── lowering/
│   ├── to-standard.mlir        [NEW]
│   ├── to-scf.mlir             [NEW]
│   ├── to-linalg.mlir          [EXTEND]
│   └── to-llvm.mlir            [NEW]
└── integration/
    ├── matmul.mlir             [EXISTS]
    ├── neural_net.mlir         [NEW]
    └── concurrent.mlir         [NEW]
```

---

## ⏱️ Implementation Timeline

### Phase 1: Core Language (Months 1-3)
**Goal**: Basic operations and control flow

**Week 1-2**: Arithmetic operations
- [ ] Add boas.add, sub, mul, div ops
- [ ] Add comparison ops
- [ ] Add tests
- [ ] Add lowering to arith dialect

**Week 3-4**: Control flow
- [ ] Add boas.if op
- [ ] Add boas.for op
- [ ] Add lowering to scf dialect
- [ ] Add tests

**Week 5-6**: Functions
- [ ] Add boas.func, call, return
- [ ] Add function type system
- [ ] Add tests

**Week 7-9**: Memory operations
- [ ] Add alloc/dealloc ops
- [ ] Add load/store ops
- [ ] Add pointer type
- [ ] Add tests

**Week 10-12**: Integration and testing
- [ ] End-to-end tests
- [ ] Performance benchmarks
- [ ] Documentation

### Phase 2: Advanced Features (Months 4-6)
**Goal**: Tensor operations and hardware acceleration

**Week 1-4**: Tensor operations
- [ ] Conv2D, Pool2D
- [ ] ReLU, Softmax
- [ ] Reshape, Transpose
- [ ] Tests and lowering

**Week 5-8**: Device operations
- [ ] Device type and ops
- [ ] To_device, execute_on
- [ ] NPU lowering (HIVM)
- [ ] GPU lowering (CUDA)

**Week 9-12**: Optimization passes
- [ ] Kernel fusion
- [ ] Device placement
- [ ] Memory optimization
- [ ] Benchmarking

### Phase 3: Concurrency (Months 7-9)
**Goal**: Async/await and channels

**Week 1-4**: Async operations
- [ ] Async/await ops
- [ ] Future type
- [ ] Runtime support

**Week 5-8**: Channel operations
- [ ] Channel type and ops
- [ ] Send/receive
- [ ] Runtime scheduler

**Week 9-12**: Integration
- [ ] Multi-device async
- [ ] Benchmarks
- [ ] Documentation

---

## 📊 Success Metrics

### Code Quality Metrics
- [ ] All ops have verifiers
- [ ] All ops have tests (>90% coverage)
- [ ] All ops have lowering passes
- [ ] Documentation complete

### Performance Metrics
- [ ] CPU performance within 10% of C++
- [ ] NPU utilization >80%
- [ ] GPU performance competitive with PyTorch

### Completeness Metrics
- [ ] Can compile basic programs end-to-end
- [ ] Can run on CPU, GPU, NPU
- [ ] Standard library 50% complete

---

## 🔗 References

- MLIR Documentation: https://mlir.llvm.org/
- TableGen Language Reference: https://llvm.org/docs/TableGen/
- Toy Tutorial: https://mlir.llvm.org/docs/Tutorials/Toy/
- Linalg Dialect: https://mlir.llvm.org/docs/Dialects/Linalg/
- SCF Dialect: https://mlir.llvm.org/docs/Dialects/SCFDialect/

---

**Document Status**: Draft v1.0
**Next Review**: After Phase 1 completion
**Owner**: Boas Language Team
