// Generated MatMul Conversion:
// This demonstrates the conversion from Boas MatMulOp
// to Linalg matmul with proper initialization.

module {
  func.func @matmul_2x3_3x4(%arg0: tensor<2x3xf32>, %arg1: tensor<3x4xf32>) -> tensor<2x4xf32> {
    %0 = tensor.empty() : tensor<2x4xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2x4xf32>) -> tensor<2x4xf32>
    %2 = linalg.matmul ins(%arg0, %arg1 : tensor<2x3xf32>, tensor<3x4xf32>) outs(%1 : tensor<2x4xf32>) -> tensor<2x4xf32>
    return %2 : tensor<2x4xf32>
  }
}


// Conversion steps:
// 1. tensor.empty() - Create output tensor
// 2. arith.constant 0.0 - Create zero constant
// 3. linalg.fill - Initialize output with zeros
// 4. linalg.matmul - Perform matrix multiplication
