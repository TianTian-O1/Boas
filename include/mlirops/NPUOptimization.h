#ifndef BOAS_NPU_OPTIMIZATION_H
#define BOAS_NPU_OPTIMIZATION_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "NPUDirectAccess.h"
#include <memory>

namespace boas {
namespace npu {

/**
 * BOAS NPU Optimization Suite
 * Combines all NPU optimizations into a single module
 */
class NPUOptimizationSuite {
public:
    // Optimization levels
    enum OptLevel {
        O0 = 0,  // No optimization
        O1 = 1,  // Basic optimizations
        O2 = 2,  // Standard optimizations (default)
        O3 = 3   // Aggressive optimizations
    };
    
    struct Config {
        OptLevel level = O2;
        bool enableMatmulOpt = true;
        bool enableMixedPrecision = true;
        bool enableSmallMatrixOpt = true;
        bool enableDirectHardware = true;
        bool enableFusion = true;
        bool enableTiling = true;
        int defaultTileSize = 64;
    };
    
    NPUOptimizationSuite(const Config& config = {});
    ~NPUOptimizationSuite();
    
    // Main optimization entry point
    void optimizeModule(mlir::ModuleOp module);
    
    // Individual optimization passes
    void optimizeMatmul(mlir::Operation* op);
    void optimizeMixedPrecision(mlir::Operation* op);
    void optimizeSmallMatrix(mlir::Operation* op);
    void optimizeWithDirectHardware(mlir::Operation* op);
    void applyFusion(mlir::ModuleOp module);
    void applyTiling(mlir::Operation* op);
    
    // Performance analysis
    struct PerfStats {
        double estimatedGFLOPS;
        size_t memoryUsage;
        int optimizationsApplied;
        bool directHardwareUsed;
    };
    
    PerfStats analyzePerformance(mlir::Operation* op);
    
private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

/**
 * NPU Matmul Optimizer
 * Optimizes matrix multiplication for Ascend NPU
 */
class NPUMatmulOptimizer {
public:
    static void optimize(mlir::Operation* op);
    static bool canOptimize(mlir::Operation* op);
    static void selectAlgorithm(mlir::Operation* op, int M, int N, int K);
    static void applyTiling(mlir::Operation* op, int tileM, int tileN, int tileK);
};

/**
 * Mixed Precision Optimizer
 * Automatic FP16/FP32 selection for optimal performance
 */
class MixedPrecisionOptimizer {
public:
    enum PrecisionMode {
        FP32,
        FP16,
        MIXED,
        AUTO
    };
    
    static void optimize(mlir::Operation* op, PrecisionMode mode = AUTO);
    static bool shouldUseFP16(int M, int N, int K);
    static void convertToFP16(mlir::Operation* op);
    static void insertCasts(mlir::Operation* op);
};

/**
 * Small Matrix Optimizer
 * Special optimizations for matrices < 256x256
 */
class SmallMatrixOptimizer {
public:
    static void optimize(mlir::Operation* op);
    static bool isSmallMatrix(mlir::Operation* op);
    static void unrollLoops(mlir::Operation* op);
    static void vectorize(mlir::Operation* op);
    static void batchSmallOperations(mlir::ModuleOp module);
};

/**
 * Fusion Optimizer
 * Fuses multiple operations for better performance
 */
class FusionOptimizer {
public:
    static void fuseMatmulChain(mlir::ModuleOp module);
    static void fuseElementwise(mlir::ModuleOp module);
    static void fuseReduction(mlir::ModuleOp module);
    static bool canFuse(mlir::Operation* op1, mlir::Operation* op2);
};

/**
 * Create optimization pass for NPU
 */
std::unique_ptr<mlir::Pass> createNPUOptimizationPass(
    const NPUOptimizationSuite::Config& config = {}
);

/**
 * Register all NPU optimization passes
 */
void registerNPUOptimizationPasses();

} // namespace npu
} // namespace boas

#endif // BOAS_NPU_OPTIMIZATION_H