//===----------------------------------------------------------------------===//
// Boas Dialect Passes Definition
//===----------------------------------------------------------------------===//

#ifndef BOAS_PASSES
#define BOAS_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// Lowering Passes
//===----------------------------------------------------------------------===//

def BoasToStandardLoweringPass : Pass<"boas-to-standard", "func::FuncOp"> {
  let summary = "Lower Boas dialect to standard dialects";
  let description = [{
    将Boas dialect的高级操作lowering到标准MLIR dialect（Arith、MemRef、SCF等）。
    
    转换包括：
    - boas.tensor.create -> memref.alloc + memref.store
    - boas.tensor.random -> runtime call + memref operations
    - boas.print -> runtime call
    
    这个pass将Boas的高级语义转换为更低级的标准操作。
  }];
  
  let constructor = "mlir::boas::createBoasToStandardLoweringPass()";
  let dependentDialects = [
    "arith::ArithDialect",
    "memref::MemRefDialect", 
    "scf::SCFDialect",
    "func::FuncDialect"
  ];
}

def BoasToLinalgLoweringPass : Pass<"boas-to-linalg", "func::FuncOp"> {
  let summary = "Lower Boas matrix operations to Linalg";
  let description = [{
    将Boas的矩阵操作转换为Linalg dialect的结构化操作。
    
    转换包括：
    - boas.matmul -> linalg.matmul 或自定义linalg.generic
    - boas.tensor -> tensor dialect operations
    
    这个pass专门处理线性代数操作的转换，为后续优化做准备。
  }];
  
  let constructor = "mlir::boas::createBoasToLinalgLoweringPass()";
  let dependentDialects = [
    "linalg::LinalgDialect",
    "tensor::TensorDialect",
    "arith::ArithDialect"
  ];
}

//===----------------------------------------------------------------------===//
// Optimization Passes  
//===----------------------------------------------------------------------===//

def BoasMatrixOptimizationPass : Pass<"boas-matrix-opt", "func::FuncOp"> {
  let summary = "Optimize Boas matrix operations";
  let description = [{
    对Boas矩阵操作进行高级优化，包括：
    
    - 矩阵运算融合（fusion）
    - 操作重排序
    - 内存访问优化
    - 常数传播和折叠
    
    这个pass在dialect级别进行优化，保持高级语义信息。
  }];
  
  let constructor = "mlir::boas::createBoasMatrixOptimizationPass()";
  let options = [
    Option<"enableFusion", "enable-fusion", "bool", /*default=*/"true",
           "Enable matrix operation fusion">,
    Option<"tileSize", "tile-size", "int64_t", /*default=*/"128",
           "Tile size for matrix operations">
  ];
}

def BoasNPUOptimizationPass : Pass<"boas-npu-opt", "func::FuncOp"> {
  let summary = "NPU-specific optimizations for Boas operations";
  let description = [{
    针对昇腾NPU的专门优化，包括：
    
    - 自动选择最优块配置（block configuration）
    - 对角线分核策略选择
    - 内存对齐优化
    - 数据类型优化（FP32 accumulator + BF16 storage）
    
    这个pass分析操作特征并应用NPU特定优化。
  }];
  
  let constructor = "mlir::boas::createBoasNPUOptimizationPass()";
  let options = [
    Option<"targetDevice", "target-device", "std::string", /*default=*/"\"npu\"",
           "Target device for optimization">,
    Option<"enableDiagonalTiling", "enable-diagonal-tiling", "bool", /*default=*/"true",
           "Enable diagonal tiling optimization">,
    Option<"blockSizeM", "block-m", "int64_t", /*default=*/"128",
           "Block size for M dimension">,
    Option<"blockSizeN", "block-n", "int64_t", /*default=*/"256", 
           "Block size for N dimension">,
    Option<"blockSizeK", "block-k", "int64_t", /*default=*/"256",
           "Block size for K dimension">
  ];
}

def DeviceAwareOptimizationPass : Pass<"boas-device-opt", "mlir::ModuleOp"> {
  let summary = "Device-aware optimization for Boas operations";
  let description = [{
    根据目标设备进行优化，包括：
    
    - 设备间数据传输优化
    - 设备特定的算子选择
    - 内存层次结构优化
    - 异构计算调度
    
    这个pass在模块级别进行跨设备优化。
  }];
  
  let constructor = "mlir::boas::createDeviceAwareOptimizationPass()";
  let options = [
    ListOption<"availableDevices", "available-devices", "std::string",
               "List of available devices", 
               "llvm::cl::MiscFlags::CommaSeparated">,
    Option<"enableAutoPlacement", "enable-auto-placement", "bool", /*default=*/"true",
           "Enable automatic device placement">
  ];
}

//===----------------------------------------------------------------------===//
// Code Generation Passes
//===----------------------------------------------------------------------===//

def NPUKernelGenerationPass : Pass<"npu-kernel-gen", "func::FuncOp"> {
  let summary = "Generate NPU kernels for Boas operations";
  let description = [{
    为Boas操作生成NPU-specific kernels，包括：
    
    - 将高级操作转换为NPU kernel调用
    - 生成kernel配置参数
    - 插入同步和内存管理代码
    - 优化kernel启动开销
    
    这个pass生成最终的NPU可执行代码。
  }];
  
  let constructor = "mlir::boas::createNPUKernelGenerationPass()";
  let options = [
    Option<"kernelNamePrefix", "kernel-prefix", "std::string", /*default=*/"\"boas_\"",
           "Prefix for generated kernel names">,
    Option<"generateDebugInfo", "debug-info", "bool", /*default=*/"false",
           "Generate debug information for kernels">,
    Option<"optimizeKernelLaunch", "optimize-launch", "bool", /*default=*/"true",
           "Optimize kernel launch overhead">
  ];
}

//===----------------------------------------------------------------------===//
// Analysis Passes
//===----------------------------------------------------------------------===//

def BoasMatrixAnalysisPass : Pass<"boas-matrix-analysis", "func::FuncOp"> {
  let summary = "Analyze Boas matrix operations for optimization opportunities";
  let description = [{
    分析Boas矩阵操作的特征，为优化提供信息：
    
    - 矩阵大小分析
    - 操作依赖关系分析  
    - 内存访问模式分析
    - 并行化机会识别
    
    这个pass不修改IR，只进行分析并输出优化建议。
  }];
  
  let constructor = "mlir::boas::createBoasMatrixAnalysisPass()";
  let options = [
    Option<"verboseOutput", "verbose", "bool", /*default=*/"false",
           "Enable verbose analysis output">,
    Option<"outputFile", "output-file", "std::string", /*default=*/"\"\"",
           "Output file for analysis results">
  ];
}

#endif // BOAS_PASSES
