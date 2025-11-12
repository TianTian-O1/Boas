//===- Passes.td - Boas conversion passes ----------------*- tablegen -*-===//
//
// Part of the Boas-NPU Project
//
//===----------------------------------------------------------------------===//

#ifndef BOAS_CONVERSION_PASSES_TD
#define BOAS_CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// BoasToLinalg
//===----------------------------------------------------------------------===//

def ConvertBoasToLinalg : Pass<"convert-boas-to-linalg", "ModuleOp"> {
  let summary = "Convert Boas dialect to Linalg dialect";
  let description = [{
    This pass converts Boas dialect operations to Linalg dialect operations.

    Conversion mappings:
    - boas.matmul -> linalg.matmul
    - boas.add -> linalg.map (with arith.addf)
    - boas.mul -> linalg.map (with arith.mulf)
    - boas.relu -> linalg.map (with arith.maxf)

    The pass also handles tensor type conversions from boas.tensor to
    standard tensor types.

    Example:
    ```mlir
    // Before:
    %C = boas.matmul %A, %B : !boas.tensor<2x3xf32>, !boas.tensor<3x4xf32>
                              -> !boas.tensor<2x4xf32>

    // After:
    %init = tensor.empty() : tensor<2x4xf32>
    %zero = arith.constant 0.0 : f32
    %C_init = linalg.fill ins(%zero : f32) outs(%init : tensor<2x4xf32>)
              -> tensor<2x4xf32>
    %C = linalg.matmul ins(%A, %B : tensor<2x3xf32>, tensor<3x4xf32>)
                        outs(%C_init : tensor<2x4xf32>) -> tensor<2x4xf32>
    ```
  }];

  let constructor = "mlir::boas::createConvertBoasToLinalgPass()";
  let dependentDialects = [
    "linalg::LinalgDialect",
    "tensor::TensorDialect",
    "arith::ArithDialect"
  ];
}

#endif // BOAS_CONVERSION_PASSES_TD
