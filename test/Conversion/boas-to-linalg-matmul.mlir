// RUN: boas-opt %s | FileCheck %s

// This file demonstrates the expected conversion from Boas MatMul to Linalg.
// Since we're still working on Boas Dialect compilation, this shows the
// target Linalg code that we want to generate.

// ============================================================================
// Test 1: Basic MatMul Conversion Pattern
// ============================================================================

// Expected input (Boas):
// %C = boas.matmul %A, %B : !boas.tensor<2x3xf32>, !boas.tensor<3x4xf32>
//                          -> !boas.tensor<2x4xf32>

// Expected output (Linalg):
func.func @matmul_2x3_3x4(%A: tensor<2x3xf32>, %B: tensor<3x4xf32>) -> tensor<2x4xf32> {
  // Step 1: Create empty output tensor
  // CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<2x4xf32>
  %empty = tensor.empty() : tensor<2x4xf32>

  // Step 2: Create zero constant for initialization
  // CHECK: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
  %zero = arith.constant 0.0 : f32

  // Step 3: Fill output with zeros (required by linalg.matmul)
  // CHECK: %[[INIT:.*]] = linalg.fill ins(%[[ZERO]] : f32) outs(%[[EMPTY]] : tensor<2x4xf32>) -> tensor<2x4xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<2x4xf32>) -> tensor<2x4xf32>

  // Step 4: Perform matrix multiplication
  // CHECK: %[[RESULT:.*]] = linalg.matmul ins(%arg0, %arg1 : tensor<2x3xf32>, tensor<3x4xf32>) outs(%[[INIT]] : tensor<2x4xf32>) -> tensor<2x4xf32>
  %result = linalg.matmul ins(%A, %B : tensor<2x3xf32>, tensor<3x4xf32>)
                           outs(%init : tensor<2x4xf32>) -> tensor<2x4xf32>

  // CHECK: return %[[RESULT]]
  return %result : tensor<2x4xf32>
}

// ============================================================================
// Test 2: Square Matrix Multiplication
// ============================================================================

func.func @matmul_square_3x3(%A: tensor<3x3xf32>, %B: tensor<3x3xf32>) -> tensor<3x3xf32> {
  // CHECK-LABEL: func @matmul_square_3x3
  // CHECK: tensor.empty() : tensor<3x3xf32>
  %empty = tensor.empty() : tensor<3x3xf32>

  // CHECK: arith.constant 0.000000e+00 : f32
  %zero = arith.constant 0.0 : f32

  // CHECK: linalg.fill
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<3x3xf32>) -> tensor<3x3xf32>

  // CHECK: linalg.matmul ins(%arg0, %arg1 : tensor<3x3xf32>, tensor<3x3xf32>) outs({{.*}} : tensor<3x3xf32>)
  %result = linalg.matmul ins(%A, %B : tensor<3x3xf32>, tensor<3x3xf32>)
                           outs(%init : tensor<3x3xf32>) -> tensor<3x3xf32>

  return %result : tensor<3x3xf32>
}

// ============================================================================
// Test 3: Large Dimension MatMul
// ============================================================================

func.func @matmul_large_128x512x256(%A: tensor<128x512xf32>, %B: tensor<512x256xf32>) -> tensor<128x256xf32> {
  // CHECK-LABEL: func @matmul_large_128x512x256
  %empty = tensor.empty() : tensor<128x256xf32>
  %zero = arith.constant 0.0 : f32
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<128x256xf32>) -> tensor<128x256xf32>

  // CHECK: linalg.matmul ins(%arg0, %arg1 : tensor<128x512xf32>, tensor<512x256xf32>)
  %result = linalg.matmul ins(%A, %B : tensor<128x512xf32>, tensor<512x256xf32>)
                           outs(%init : tensor<128x256xf32>) -> tensor<128x256xf32>

  return %result : tensor<128x256xf32>
}

// ============================================================================
// Test 4: Conversion Helper Function Demonstration
// ============================================================================

// This demonstrates the conversion logic that would be in the pass
func.func @conversion_example(%A: tensor<4x5xf32>, %B: tensor<5x6xf32>) -> tensor<4x6xf32> {
  // The convertMatMulOp function in BoasToLinalg.cpp generates this pattern:

  // 1. Create empty tensor with result shape
  %c4 = arith.constant 4 : index
  %c6 = arith.constant 6 : index
  %empty = tensor.empty() : tensor<4x6xf32>

  // 2. Create zero constant
  %zero = arith.constant 0.0 : f32

  // 3. Initialize output
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<4x6xf32>) -> tensor<4x6xf32>

  // 4. Compute matmul
  %result = linalg.matmul ins(%A, %B : tensor<4x5xf32>, tensor<5x6xf32>)
                           outs(%init : tensor<4x6xf32>) -> tensor<4x6xf32>

  return %result : tensor<4x6xf32>
}

// ============================================================================
// Test 5: f64 Element Type
// ============================================================================

func.func @matmul_f64(%A: tensor<2x2xf64>, %B: tensor<2x2xf64>) -> tensor<2x2xf64> {
  // CHECK-LABEL: func @matmul_f64
  %empty = tensor.empty() : tensor<2x2xf64>

  // For f64, we need f64 zero constant
  // CHECK: arith.constant 0.000000e+00 : f64
  %zero = arith.constant 0.0 : f64

  %init = linalg.fill ins(%zero : f64) outs(%empty : tensor<2x2xf64>) -> tensor<2x2xf64>

  // CHECK: linalg.matmul ins(%arg0, %arg1 : tensor<2x2xf64>, tensor<2x2xf64>)
  %result = linalg.matmul ins(%A, %B : tensor<2x2xf64>, tensor<2x2xf64>)
                           outs(%init : tensor<2x2xf64>) -> tensor<2x2xf64>

  return %result : tensor<2x2xf64>
}
