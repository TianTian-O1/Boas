//===- BoasOps_minimal.td - Minimal Boas ops for MatMul ----*- tablegen -*-===//
//
// Minimal version focusing only on MatMul to avoid LLVM 20 API issues
//
//===----------------------------------------------------------------------===//

#ifndef BOAS_DIALECT_BOAS_IR_BOASOPS_MINIMAL_TD
#define BOAS_DIALECT_BOAS_IR_BOASOPS_MINIMAL_TD

include "Boas/Dialect/Boas/IR/BoasDialect.td"
include "Boas/Dialect/Boas/IR/BoasTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

//===----------------------------------------------------------------------===//
// MatMul Operation (Core功能)
//===----------------------------------------------------------------------===//

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

#endif // BOAS_DIALECT_BOAS_IR_BOASOPS_MINIMAL_TD
