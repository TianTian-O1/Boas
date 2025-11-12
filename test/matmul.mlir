// RUN: boas-opt %s | boas-opt | FileCheck %s

// -----
// Test 1: Basic 2D matrix multiplication
// -----

module {
  // CHECK-LABEL: func @test_matmul_2d
  boas.func @test_matmul_2d() -> !boas.tensor<2x4xf32> {
    // Create input tensors
    // CHECK: %[[A:.*]] = boas.tensor.create
    %A = boas.tensor.create dense<[[1.0, 2.0, 3.0],
                                     [4.0, 5.0, 6.0]]>
         : !boas.tensor<2x3xf32>

    // CHECK: %[[B:.*]] = boas.tensor.create
    %B = boas.tensor.create dense<[[1.0, 2.0, 3.0, 4.0],
                                     [5.0, 6.0, 7.0, 8.0],
                                     [9.0, 10.0, 11.0, 12.0]]>
         : !boas.tensor<3x4xf32>

    // Matrix multiplication: C = A * B
    // A: 2x3, B: 3x4 -> C: 2x4
    // CHECK: %[[C:.*]] = boas.matmul %[[A]], %[[B]] : !boas.tensor<2x3xf32>, !boas.tensor<3x4xf32> -> !boas.tensor<2x4xf32>
    %C = boas.matmul %A, %B : !boas.tensor<2x3xf32>,
                               !boas.tensor<3x4xf32>
                            -> !boas.tensor<2x4xf32>

    // CHECK: boas.return %[[C]]
    boas.return %C : !boas.tensor<2x4xf32>
  }
}

// -----
// Test 2: Square matrix multiplication
// -----

module {
  // CHECK-LABEL: func @test_matmul_square
  boas.func @test_matmul_square() -> !boas.tensor<3x3xf32> {
    // CHECK: %[[A:.*]] = boas.tensor.create
    %A = boas.tensor.create dense<[[1.0, 2.0, 3.0],
                                     [4.0, 5.0, 6.0],
                                     [7.0, 8.0, 9.0]]>
         : !boas.tensor<3x3xf32>

    // CHECK: %[[B:.*]] = boas.tensor.create
    %B = boas.tensor.create dense<[[9.0, 8.0, 7.0],
                                     [6.0, 5.0, 4.0],
                                     [3.0, 2.0, 1.0]]>
         : !boas.tensor<3x3xf32>

    // CHECK: %[[C:.*]] = boas.matmul %[[A]], %[[B]] : !boas.tensor<3x3xf32>, !boas.tensor<3x3xf32> -> !boas.tensor<3x3xf32>
    %C = boas.matmul %A, %B : !boas.tensor<3x3xf32>,
                               !boas.tensor<3x3xf32>
                            -> !boas.tensor<3x3xf32>

    // CHECK: boas.return %[[C]]
    boas.return %C : !boas.tensor<3x3xf32>
  }
}

// -----
// Test 3: Large dimensions
// -----

module {
  // CHECK-LABEL: func @test_matmul_large
  boas.func @test_matmul_large() -> !boas.tensor<128x256xf32> {
    // CHECK: %[[A:.*]] = boas.tensor.random
    %A = boas.tensor.random : !boas.tensor<128x512xf32>

    // CHECK: %[[B:.*]] = boas.tensor.random
    %B = boas.tensor.random : !boas.tensor<512x256xf32>

    // CHECK: %[[C:.*]] = boas.matmul %[[A]], %[[B]] : !boas.tensor<128x512xf32>, !boas.tensor<512x256xf32> -> !boas.tensor<128x256xf32>
    %C = boas.matmul %A, %B : !boas.tensor<128x512xf32>,
                               !boas.tensor<512x256xf32>
                            -> !boas.tensor<128x256xf32>

    // CHECK: boas.return %[[C]]
    boas.return %C : !boas.tensor<128x256xf32>
  }
}

// -----
// Test 4: NPU accelerated matrix multiplication
// -----

module {
  // CHECK-LABEL: func @test_matmul_npu
  boas.func @test_matmul_npu() -> !boas.tensor<1024x1024xf32> {
    // Get NPU device
    // CHECK: %[[DEV:.*]] = boas.get_device
    %device = boas.get_device 0 : !boas.device<0>

    // Create random tensors
    // CHECK: %[[A:.*]] = boas.tensor.random
    %A = boas.tensor.random : !boas.tensor<1024x1024xf32>
    // CHECK: %[[B:.*]] = boas.tensor.random
    %B = boas.tensor.random : !boas.tensor<1024x1024xf32>

    // Move to NPU
    // CHECK: %[[A_NPU:.*]] = boas.to_device %[[A]], %[[DEV]]
    %A_npu = boas.to_device %A, %device : !boas.tensor<1024x1024xf32>, !boas.device<0>
    // CHECK: %[[B_NPU:.*]] = boas.to_device %[[B]], %[[DEV]]
    %B_npu = boas.to_device %B, %device : !boas.tensor<1024x1024xf32>, !boas.device<0>

    // MatMul on NPU
    // CHECK: %[[C_NPU:.*]] = boas.matmul %[[A_NPU]], %[[B_NPU]]
    %C_npu = boas.matmul %A_npu, %B_npu : !boas.tensor<1024x1024xf32>,
                                           !boas.tensor<1024x1024xf32>
                                        -> !boas.tensor<1024x1024xf32>

    // CHECK: boas.return %[[C_NPU]]
    boas.return %C_npu : !boas.tensor<1024x1024xf32>
  }
}
