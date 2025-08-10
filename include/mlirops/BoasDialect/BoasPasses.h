#ifndef BOAS_PASSES_H
#define BOAS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace boas {

//===----------------------------------------------------------------------===//
// Boas Dialect Passes
//===----------------------------------------------------------------------===//

/// 创建Boas到标准dialect的lowering pass
std::unique_ptr<Pass> createBoasToStandardLoweringPass();

/// 创建Boas矩阵操作优化pass
std::unique_ptr<Pass> createBoasMatrixOptimizationPass();

/// 创建NPU特定优化pass
std::unique_ptr<Pass> createBoasNPUOptimizationPass();

/// 创建Boas到Linalg的lowering pass
std::unique_ptr<Pass> createBoasToLinalgLoweringPass();

/// 创建NPU kernel生成pass
std::unique_ptr<Pass> createNPUKernelGenerationPass();

/// 创建设备感知优化pass
std::unique_ptr<Pass> createDeviceAwareOptimizationPass();

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

/// 注册所有Boas passes
void registerBoasPasses();

#define GEN_PASS_CLASSES
#include "mlirops/BoasDialect/BoasPasses.h.inc"

} // namespace boas
} // namespace mlir

#endif // BOAS_PASSES_H
