// Example: Matrix Multiplication in Linalg
// This represents the output of: boas.matmul lowered to Linalg
//
// Computes: C = A * B where A is 2x3, B is 3x4, C is 2x4
//
// This file can be executed with: boas-run matmul_simple.mlir

module {
  // Matrix multiplication function: C[2x4] = A[2x3] * B[3x4]
  func.func @matmul_example(%A: tensor<2x3xf32>, %B: tensor<3x4xf32>) -> tensor<2x4xf32> {
    // Step 1: Create empty output tensor
    %empty = tensor.empty() : tensor<2x4xf32>

    // Step 2: Create zero constant for initialization
    %zero = arith.constant 0.0 : f32

    // Step 3: Initialize output tensor with zeros
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<2x4xf32>) -> tensor<2x4xf32>

    // Step 4: Perform matrix multiplication
    // C[i,j] = sum_k(A[i,k] * B[k,j])
    %result = linalg.matmul
      ins(%A, %B : tensor<2x3xf32>, tensor<3x4xf32>)
      outs(%init : tensor<2x4xf32>) -> tensor<2x4xf32>

    return %result : tensor<2x4xf32>
  }

  // Main function that calls the matmul with concrete values
  func.func @main() -> i32 {
    // Create test input matrices
    // A = [[1.0, 2.0, 3.0],
    //      [4.0, 5.0, 6.0]]
    %A_values = arith.constant dense<[[1.0, 2.0, 3.0],
                                        [4.0, 5.0, 6.0]]> : tensor<2x3xf32>

    // B = [[1.0, 2.0, 3.0, 4.0],
    //      [5.0, 6.0, 7.0, 8.0],
    //      [9.0, 10.0, 11.0, 12.0]]
    %B_values = arith.constant dense<[[1.0, 2.0, 3.0, 4.0],
                                        [5.0, 6.0, 7.0, 8.0],
                                        [9.0, 10.0, 11.0, 12.0]]> : tensor<3x4xf32>

    // Call matmul function
    %result = func.call @matmul_example(%A_values, %B_values) :
      (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>

    // Expected result:
    // C = [[38.0, 44.0, 50.0, 56.0],
    //      [83.0, 98.0, 113.0, 128.0]]

    // Print result (when execution engine supports it)
    // For now, just return success
    %success = arith.constant 0 : i32
    return %success : i32
  }
}
