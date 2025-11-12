// Minimal matrix multiplication example
// This is the exact output of our Boas→Linalg lowering pass

module {
  func.func @matmul(%arg0: tensor<2x3xf32>, %arg1: tensor<3x4xf32>) -> tensor<2x4xf32> {
    %0 = tensor.empty() : tensor<2x4xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2x4xf32>) -> tensor<2x4xf32>
    %2 = linalg.matmul ins(%arg0, %arg1 : tensor<2x3xf32>, tensor<3x4xf32>)
                        outs(%1 : tensor<2x4xf32>) -> tensor<2x4xf32>
    return %2 : tensor<2x4xf32>
  }
}
