module {
  func.func private @printMemrefF64(memref<*xf64>)

  func.func @main() {
    // 创建一个简单的 2x2 矩阵
    %matrix = memref.alloc() : memref<2x2xf64>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %val = arith.constant 42.0 : f64
    
    // 存储一些值
    memref.store %val, %matrix[%c0, %c0] : memref<2x2xf64>
    memref.store %val, %matrix[%c0, %c1] : memref<2x2xf64>
    memref.store %val, %matrix[%c1, %c0] : memref<2x2xf64>
    memref.store %val, %matrix[%c1, %c1] : memref<2x2xf64>
    
    // 转换为通用 memref 类型并打印
    %cast = memref.cast %matrix : memref<2x2xf64> to memref<*xf64>
    call @printMemrefF64(%cast) : (memref<*xf64>) -> ()
    
    // 释放内存
    memref.dealloc %matrix : memref<2x2xf64>
    return
  }
}