// Boas Dialect 示例MLIR文件
// 展示自定义dialect的语法和语义

module {
  // 使用Boas dialect进行矩阵乘法
  func.func @boas_matmul_example() -> !boas.tensor<1024x1024xbf16@npu> {
    %c1024 = arith.constant 1024 : index
    
    // 创建NPU上的随机张量
    %A = boas.tensor.random(%c1024, %c1024) {device = "npu"}
        : !boas.tensor<1024x1024xbf16@npu>
    %B = boas.tensor.random(%c1024, %c1024) {device = "npu"}
        : !boas.tensor<1024x1024xbf16@npu>
    
    // NPU优化的矩阵乘法
    %C = boas.matmul %A, %B {
        npu_opt = #boas.npu_opt<128, 256, 256, true, "diagonal">
    } : (!boas.tensor<1024x1024xbf16@npu>, !boas.tensor<1024x1024xbf16@npu>) 
      -> !boas.tensor<1024x1024xbf16@npu>
    
    return %C : !boas.tensor<1024x1024xbf16@npu>
  }
  
  // 演示设备转移
  func.func @device_transfer_example() {
    %c512 = arith.constant 512 : index
    
    // CPU张量
    %cpu_tensor = boas.tensor.random(%c512, %c512) {device = "cpu"}
        : !boas.tensor<512x512xf32@cpu>
    
    // 转移到NPU
    %npu_tensor = boas.to_device %cpu_tensor to "npu"
        : !boas.tensor<512x512xf32@cpu> -> !boas.tensor<512x512xf32@npu>
    
    return
  }
  
  // 演示NPU kernel操作
  func.func @npu_kernel_example() {
    %c256 = arith.constant 256 : index
    
    %A = boas.tensor.random(%c256, %c256) {device = "npu"}
        : !boas.tensor<256x256xf32@npu>
    %B = boas.tensor.random(%c256, %c256) {device = "npu"}
        : !boas.tensor<256x256xf32@npu>
    
    // 直接NPU kernel调用
    %result = boas.npu.kernel %A, %B -> %C
        kernel "boas_optimized_matmul"
        config #boas.npu_opt<64, 128, 128, false, "sequential"> {
        ^bb0(%arg0: !boas.tensor<256x256xf32@npu>, 
             %arg1: !boas.tensor<256x256xf32@npu>):
          // kernel body会由CodeGen自动生成
          boas.npu.launch grid (10, 1, 1) block (1, 1, 1) {
            // NPU特定计算逻辑
          }
    } : (!boas.tensor<256x256xf32@npu>, !boas.tensor<256x256xf32@npu>) 
      -> !boas.tensor<256x256xf32@npu>
    
    return
  }
}
