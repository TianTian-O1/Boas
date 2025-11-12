// RUN: boas-opt %s | FileCheck %s

// Simple matmul test using standard MLIR dialects
module {
  func.func @matmul_test(%A: tensor<2x3xf32>, %B: tensor<3x4xf32>) -> tensor<2x4xf32> {
    // Note: Using linalg.matmul directly as a baseline
    // Later we'll test boas.matmul once it's fully working

    %empty = tensor.empty() : tensor<2x4xf32>
    %zero = arith.constant 0.0 : f32
    %C_init = linalg.fill ins(%zero : f32) outs(%empty : tensor<2x4xf32>) -> tensor<2x4xf32>

    // CHECK: linalg.matmul
    %C = linalg.matmul ins(%A, %B : tensor<2x3xf32>, tensor<3x4xf32>)
                        outs(%C_init : tensor<2x4xf32>) -> tensor<2x4xf32>

    return %C : tensor<2x4xf32>
  }
}
