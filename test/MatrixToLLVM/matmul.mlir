"builtin.module"() ({
  func.func @test_matmul() {
    %A = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf64>
    %B = arith.constant dense<[[5.0, 6.0], [7.0, 8.0]]> : tensor<2x2xf64>
    
    // 执行矩阵乘法
    %C = matrix.matmul %A, %B : tensor<2x2xf64>
    
  }
}) : () -> ()