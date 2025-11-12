//===- BoasOps.td - Boas operation definitions ------------*- tablegen -*-===//
//
// Part of the Boas-NPU Project
//
//===----------------------------------------------------------------------===//

#ifndef BOAS_DIALECT_BOAS_IR_BOASOPS_TD
#define BOAS_DIALECT_BOAS_IR_BOASOPS_TD

include "Boas/Dialect/Boas/IR/BoasDialect.td"
include "Boas/Dialect/Boas/IR/BoasTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/OpAsmInterface.td"

//===----------------------------------------------------------------------===//
// Tensor Operations
//===----------------------------------------------------------------------===//

def Boas_TensorCreateOp : Boas_Op<"tensor.create",
    [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Create a tensor with given shape and initial values";
  let description = [{
    Creates a new tensor with the specified shape and optionally initializes
    it with the given values.

    Example:
    ```mlir
    // Create 2x3 tensor with specified values
    %0 = boas.tensor.create dense<[[1.0, 2.0, 3.0],
                                     [4.0, 5.0, 6.0]]> : tensor<2x3xf32>

    // Create empty tensor
    %1 = boas.tensor.create : !boas.tensor<4x5xf32>
    ```
  }];

  let arguments = (ins
    OptionalAttr<AnyAttr>:$value
  );

  let results = (outs Boas_Tensor:$result);

  let builders = [
    OpBuilder<(ins "Type":$resultType), [{
      build($_builder, $_state, resultType, nullptr);
    }]>,
    OpBuilder<(ins "Type":$resultType, "Attribute":$value), [{
      build($_builder, $_state, resultType, value);
    }]>
  ];

  // Use default assembly format for now
}

def Boas_TensorRandomOp : Boas_Op<"tensor.random", [Pure]> {
  let summary = "Create a tensor with random values";
  let description = [{
    Creates a tensor filled with random values.

    Example:
    ```mlir
    %0 = boas.tensor.random : !boas.tensor<128x128xf32>
    ```
  }];

  let arguments = (ins);
  let results = (outs Boas_Tensor:$result);

  let assemblyFormat = "attr-dict `:` type($result)";
}

def Boas_MatMulOp : Boas_Op<"matmul",
    [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Matrix multiplication";
  let description = [{
    Performs matrix multiplication of two tensors.
    Supports 2D matrix multiplication: C = A * B

    Example:
    ```mlir
    %C = boas.matmul %A, %B : !boas.tensor<2x3xf32>, !boas.tensor<3x4xf32>
                              -> !boas.tensor<2x4xf32>
    ```
  }];

  let arguments = (ins
    Boas_Tensor:$lhs,
    Boas_Tensor:$rhs
  );

  let results = (outs Boas_Tensor:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)
  }];

  let hasVerifier = 1;
}

def Boas_AddOp : Boas_Op<"add", [Pure, Commutative, SameOperandsAndResultType]> {
  let summary = "Element-wise tensor addition";
  let description = [{
    Performs element-wise addition of two tensors.

    Example:
    ```mlir
    %result = boas.add %lhs, %rhs : !boas.tensor<2x3xf32>
    ```
  }];

  let arguments = (ins
    Boas_Tensor:$lhs,
    Boas_Tensor:$rhs
  );

  let results = (outs Boas_Tensor:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs)";
}

def Boas_MulOp : Boas_Op<"mul", [Pure, Commutative, SameOperandsAndResultType]> {
  let summary = "Element-wise tensor multiplication";
  let description = [{
    Performs element-wise multiplication of two tensors.

    Example:
    ```mlir
    %result = boas.mul %lhs, %rhs : !boas.tensor<2x3xf32>
    ```
  }];

  let arguments = (ins
    Boas_Tensor:$lhs,
    Boas_Tensor:$rhs
  );

  let results = (outs Boas_Tensor:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs)";
}

def Boas_ReluOp : Boas_Op<"relu", [Pure, SameOperandsAndResultType]> {
  let summary = "ReLU activation function";
  let description = [{
    Applies ReLU activation function element-wise: max(0, x)

    Example:
    ```mlir
    %result = boas.relu %input : !boas.tensor<2x3xf32>
    ```
  }];

  let arguments = (ins Boas_Tensor:$input);
  let results = (outs Boas_Tensor:$result);

  let assemblyFormat = "$input attr-dict `:` type($input)";
}

//===----------------------------------------------------------------------===//
// NPU Device Operations
//===----------------------------------------------------------------------===//

def Boas_GetDeviceOp : Boas_Op<"get_device", []> {
  let summary = "Get NPU device handle";
  let description = [{
    Retrieves a handle to the NPU device with the specified ID.

    Example:
    ```mlir
    %device = boas.get_device 0 : !boas.device<0>
    ```
  }];

  let arguments = (ins I32Attr:$device_id);
  let results = (outs Boas_Device:$device);

  let assemblyFormat = "$device_id attr-dict `:` type($device)";
}

def Boas_ToDeviceOp : Boas_Op<"to_device", []> {
  let summary = "Move tensor to NPU device";
  let description = [{
    Moves a tensor to the specified NPU device.

    Example:
    ```mlir
    %gpu_tensor = boas.to_device %tensor, %device
                  : !boas.tensor<2x3xf32>, !boas.device<0>
    ```
  }];

  let arguments = (ins
    Boas_Tensor:$tensor,
    Boas_Device:$device
  );

  let results = (outs Boas_Tensor:$result);

  let assemblyFormat = [{
    $tensor `,` $device attr-dict `:` type($tensor) `,` type($device) `->` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// I/O Operations
//===----------------------------------------------------------------------===//

def Boas_PrintOp : Boas_Op<"print", []> {
  let summary = "Print a value to console";
  let description = [{
    Prints the given value to console.

    Example:
    ```mlir
    boas.print %tensor : !boas.tensor<2x3xf32>
    ```
  }];

  let arguments = (ins AnyType:$value);

  let assemblyFormat = "$value attr-dict `:` type($value)";
}

//===----------------------------------------------------------------------===//
// Function Operations
//===----------------------------------------------------------------------===//

// Note: FuncOp temporarily commented out to use standard mlir.func
// Will be re-enabled after resolving LLVM 20 API compatibility

/*
def Boas_FuncOp : Boas_Op<"func", [
    IsolatedFromAbove,
    FunctionOpInterface
  ]> {
  let summary = "Boas function definition";
  let description = [{
    Defines a function in Boas language.

    Example:
    ```mlir
    boas.func @matmul(%A: !boas.tensor<2x3xf32>, %B: !boas.tensor<3x4xf32>)
        -> !boas.tensor<2x4xf32> {
      %C = boas.matmul %A, %B : !boas.tensor<2x3xf32>, !boas.tensor<3x4xf32>
                               -> !boas.tensor<2x4xf32>
      boas.return %C : !boas.tensor<2x4xf32>
    }
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );

  let regions = (region AnyRegion:$body);

  let builders = [
    OpBuilder<(ins
      "StringRef":$name,
      "FunctionType":$type,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>
  ];

  let extraClassDeclaration = [{
    // FunctionOpInterface methods
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    // Get function body
    Region *getCallableRegion() { return &getBody(); }
  }];

  let hasCustomAssemblyFormat = 1;
}

def Boas_ReturnOp : Boas_Op<"return", [
    Pure,
    HasParent<"FuncOp">,
    ReturnLike,
    Terminator
  ]> {
  let summary = "Return from function";
  let description = [{
    Returns values from a Boas function.

    Example:
    ```mlir
    boas.return %result : !boas.tensor<2x4xf32>
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [
    OpBuilder<(ins), [{
      build($_builder, $_state, llvm::None);
    }]>
  ];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}
*/

//===----------------------------------------------------------------------===//
// Fusion Hint Operations
//===----------------------------------------------------------------------===//

def Boas_FusionRegionOp : Boas_Op<"fusion_region", [
    SingleBlock,
    NoTerminator
  ]> {
  let summary = "Hint for operator fusion";
  let description = [{
    Marks a region of operations that should be fused together for
    better NPU performance.

    Example:
    ```mlir
    %result = boas.fusion_region {
      %0 = boas.add %A, %B : !boas.tensor<2x3xf32>
      %1 = boas.relu %0 : !boas.tensor<2x3xf32>
      boas.yield %1 : !boas.tensor<2x3xf32>
    } : !boas.tensor<2x3xf32>
    ```
  }];

  let regions = (region SizedRegion<1>:$body);
  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat = "$body attr-dict `:` type($results)";
}

def Boas_YieldOp : Boas_Op<"yield", [
    Pure,
    ReturnLike,
    Terminator,
    HasParent<"FusionRegionOp">
  ]> {
  let summary = "Yield values from fusion region";
  let description = [{
    Yields values from a fusion region.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

#endif // BOAS_DIALECT_BOAS_IR_BOASOPS_TD
