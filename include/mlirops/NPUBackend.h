#ifndef BOAS_NPU_BACKEND_H
#define BOAS_NPU_BACKEND_H

#include "mlirops/MLIRGen.h"
#include <memory>
#include <string>

namespace matrix {

/// NPU Backend for Ascend devices
class NPUBackend {
public:
    /// Initialize NPU backend with device configuration
    static bool initialize();
    
    /// Check if NPU devices are available
    static bool isAvailable();
    
    /// Get number of available NPU devices
    static int getDeviceCount();
    
    /// Set current NPU device
    static bool setDevice(int deviceId);
    
    /// Get current NPU device properties
    static std::string getDeviceProperties();
    
    /// Generate NPU-optimized MLIR for matrix multiplication
    /// Uses Triton-style kernels for high performance
    static mlir::Value generateNPUMatmul(
        MLIRGen* generator,
        mlir::Value lhs, 
        mlir::Value rhs,
        mlir::Value M, 
        mlir::Value N, 
        mlir::Value K
    );
    
    /// Generate Triton kernel for matrix multiplication
    /// Implements NPU-optimized block strategies
    static std::string generateTritonMatmulKernel(
        int blockM = 128,
        int blockN = 256, 
        int blockK = 256,
        bool enableDiagonalTiling = true
    );
    
private:
    static bool initialized_;
    static int currentDevice_;
    static int deviceCount_;
};

/// NPU-specific matrix multiplication optimizer
class NPUMatmulOptimizer {
public:
    /// Analyze matrix dimensions and choose optimal block size
    struct BlockConfig {
        int blockM;
        int blockN; 
        int blockK;
        bool useDiagonalTiling;
        int numCores;
    };
    
    /// Get optimal configuration for given matrix dimensions
    static BlockConfig getOptimalConfig(int M, int N, int K);
    
    /// Generate NPU-optimized kernel code
    static std::string generateOptimizedKernel(const BlockConfig& config);
    
private:
    /// NPU芯片更加亲和512B对齐场景的推荐配置
    static constexpr int DEFAULT_BLOCK_M = 128;
    static constexpr int DEFAULT_BLOCK_N = 256;
    static constexpr int DEFAULT_BLOCK_K = 256;
    
    /// 对角线分核阈值：当任务量较多时启用
    static constexpr int DIAGONAL_TILING_THRESHOLD = 8;
};

} // namespace matrix

#endif // BOAS_NPU_BACKEND_H
