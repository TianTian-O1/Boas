#ifndef BOAS_GPU_BACKEND_H
#define BOAS_GPU_BACKEND_H

#include "mlirops/MLIRGen.h"
#include <memory>
#include <string>

namespace matrix {

/// GPU Backend for Nvidia CUDA devices
class GPUBackend {
public:
    /// Initialize GPU backend with device configuration
    static bool initialize();

    /// Check if GPU devices are available
    static bool isAvailable();

    /// Get number of available GPU devices
    static int getDeviceCount();

    /// Set current GPU device
    static bool setDevice(int deviceId);

    /// Get current GPU device properties
    static std::string getDeviceProperties();

    /// Generate GPU-optimized MLIR for matrix multiplication
    static mlir::Value generateGPUMatmul(
        MLIRGen* generator,
        mlir::Value lhs,
        mlir::Value rhs,
        mlir::Value M,
        mlir::Value N,
        mlir::Value K
    );

    /// Generate CUDA kernel for matrix multiplication
    /// Implements GPU-optimized tiling strategies
    static std::string generateCUDAMatmulKernel(
        int blockM = 32,
        int blockN = 32,
        int blockK = 32,
        bool enableTensorCore = true
    );

private:
    static bool initialized_;
    static int currentDevice_;
    static int deviceCount_;
};

/// GPU-specific matrix multiplication optimizer
class GPUMatmulOptimizer {
public:
    /// Analyze matrix dimensions and choose optimal block size
    struct BlockConfig {
        int blockM;
        int blockN;
        int blockK;
        bool useTensorCore;
        int threadsPerBlock;
        int smCount;
    };

    /// Get optimal configuration for given matrix dimensions
    static BlockConfig getOptimalConfig(int M, int N, int K);

    /// Generate GPU-optimized kernel code
    static std::string generateOptimizedKernel(const BlockConfig& config);

private:
    /// GPU推荐的分块大小（针对Tensor Core优化）
    static constexpr int DEFAULT_BLOCK_M = 32;
    static constexpr int DEFAULT_BLOCK_N = 32;
    static constexpr int DEFAULT_BLOCK_K = 32;

    /// Tensor Core需要16的倍数对齐
    static constexpr int TENSOR_CORE_ALIGNMENT = 16;

    /// 使用Tensor Core的最小矩阵维度
    static constexpr int TENSOR_CORE_MIN_DIM = 128;
};

} // namespace matrix

#endif // BOAS_GPU_BACKEND_H
