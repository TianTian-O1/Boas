#include "mlirops/GPUBackend.h"
#include "mlirops/CUDARuntime.h"
#include <iostream>
#include <sstream>
#include <cmath>

namespace matrix {

// Static member initialization
bool GPUBackend::initialized_ = false;
int GPUBackend::currentDevice_ = 0;
int GPUBackend::deviceCount_ = 0;

bool GPUBackend::initialize() {
    if (initialized_) return true;

    std::cout << "[GPU] Initializing Nvidia CUDA backend..." << std::endl;

    // 使用CUDA运行时初始化
    auto& cudaRuntime = CUDARuntime::getInstance();
    if (!cudaRuntime.initialize()) {
        std::cerr << "[GPU] Failed to initialize CUDA runtime: " << cudaRuntime.getLastError() << std::endl;
        return false;
    }

    deviceCount_ = cudaRuntime.getDeviceCount();
    currentDevice_ = cudaRuntime.getCurrentDevice();
    initialized_ = true;

    std::cout << "[GPU] Successfully initialized with " << deviceCount_ << " device(s)" << std::endl;
    std::cout << "[GPU] Current device: " << cudaRuntime.getDeviceProperties(currentDevice_) << std::endl;
    return true;
}

bool GPUBackend::isAvailable() {
    // 检查CUDA运行时是否可用
    auto& cudaRuntime = CUDARuntime::getInstance();
    if (!cudaRuntime.isAvailable()) {
        // 尝试初始化
        if (!initialize()) {
            return false;
        }
    }
    return cudaRuntime.isAvailable();
}

int GPUBackend::getDeviceCount() {
    if (!initialized_) {
        initialize();
    }
    return deviceCount_;
}

bool GPUBackend::setDevice(int deviceId) {
    if (deviceId < 0 || deviceId >= deviceCount_) {
        std::cerr << "[GPU] Invalid device ID: " << deviceId << std::endl;
        return false;
    }

    auto& cudaRuntime = CUDARuntime::getInstance();
    if (!cudaRuntime.setDevice(deviceId)) {
        return false;
    }

    currentDevice_ = deviceId;
    std::cout << "[GPU] Set current device to " << deviceId << std::endl;
    return true;
}

std::string GPUBackend::getDeviceProperties() {
    auto& cudaRuntime = CUDARuntime::getInstance();
    return cudaRuntime.getDeviceProperties(currentDevice_);
}

mlir::Value GPUBackend::generateGPUMatmul(
    MLIRGen* generator,
    mlir::Value lhs,
    mlir::Value rhs,
    mlir::Value M,
    mlir::Value N,
    mlir::Value K) {

    std::cout << "[GPU] Generating CUDA-optimized matrix multiplication" << std::endl;

    auto loc = generator->getBuilder()->getUnknownLoc();

    // Create result tensor
    auto resultType = mlir::MemRefType::get(
        {mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic},
        generator->getBuilder()->getF64Type()
    );

    auto result = generator->getBuilder()->create<mlir::memref::AllocOp>(
        loc, resultType, mlir::ValueRange{M, N}
    );

    // 生成GPU优化的linalg.matmul，添加特殊属性标记
    auto matmulOp = generator->getBuilder()->create<mlir::linalg::MatmulOp>(
        loc,
        mlir::ValueRange{lhs, rhs},
        mlir::ValueRange{result}
    );

    // 添加GPU优化属性
    auto stringAttr = generator->getBuilder()->getStringAttr("gpu_optimized");
    matmulOp->setAttr("boas.backend", stringAttr);

    // 添加设备属性
    auto deviceAttr = generator->getBuilder()->getStringAttr("nvidia_gpu");
    matmulOp->setAttr("boas.device", deviceAttr);

    // 添加优化策略属性
    auto strategyAttr = generator->getBuilder()->getStringAttr("cuda_matmul");
    matmulOp->setAttr("boas.strategy", strategyAttr);

    std::cout << "[GPU] Generated CUDA-optimized linalg.matmul with GPU attributes" << std::endl;
    return result;
}

std::string GPUBackend::generateCUDAMatmulKernel(
    int blockM,
    int blockN,
    int blockK,
    bool enableTensorCore) {

    std::stringstream kernel;

    kernel << "// GPU-optimized CUDA matrix multiplication kernel\n";
    kernel << "// Block sizes: M=" << blockM << ", N=" << blockN << ", K=" << blockK << "\n";
    kernel << "// Tensor Core: " << (enableTensorCore ? "enabled" : "disabled") << "\n\n";

    kernel << "__global__ void gpu_matmul_kernel(\n";
    kernel << "    const float* __restrict__ A,\n";
    kernel << "    const float* __restrict__ B,\n";
    kernel << "    float* __restrict__ C,\n";
    kernel << "    int M, int N, int K) {\n";
    kernel << "    \n";
    kernel << "    // Shared memory for tile caching\n";
    kernel << "    __shared__ float As[" << blockM << "][" << blockK << "];\n";
    kernel << "    __shared__ float Bs[" << blockK << "][" << blockN << "];\n";
    kernel << "    \n";
    kernel << "    // Thread indices\n";
    kernel << "    int tx = threadIdx.x;\n";
    kernel << "    int ty = threadIdx.y;\n";
    kernel << "    int bx = blockIdx.x;\n";
    kernel << "    int by = blockIdx.y;\n";
    kernel << "    \n";
    kernel << "    // Calculate global position\n";
    kernel << "    int row = by * " << blockM << " + ty;\n";
    kernel << "    int col = bx * " << blockN << " + tx;\n";
    kernel << "    \n";
    kernel << "    float sum = 0.0f;\n";
    kernel << "    \n";
    kernel << "    // Tile over K dimension\n";
    kernel << "    for (int t = 0; t < (K + " << blockK << " - 1) / " << blockK << "; t++) {\n";
    kernel << "        // Load tiles into shared memory\n";
    kernel << "        if (row < M && t * " << blockK << " + tx < K)\n";
    kernel << "            As[ty][tx] = A[row * K + t * " << blockK << " + tx];\n";
    kernel << "        else\n";
    kernel << "            As[ty][tx] = 0.0f;\n";
    kernel << "        \n";
    kernel << "        if (col < N && t * " << blockK << " + ty < K)\n";
    kernel << "            Bs[ty][tx] = B[(t * " << blockK << " + ty) * N + col];\n";
    kernel << "        else\n";
    kernel << "            Bs[ty][tx] = 0.0f;\n";
    kernel << "        \n";
    kernel << "        __syncthreads();\n";
    kernel << "        \n";
    kernel << "        // Compute partial dot product\n";
    kernel << "        #pragma unroll\n";
    kernel << "        for (int k = 0; k < " << blockK << "; k++) {\n";
    kernel << "            sum += As[ty][k] * Bs[k][tx];\n";
    kernel << "        }\n";
    kernel << "        \n";
    kernel << "        __syncthreads();\n";
    kernel << "    }\n";
    kernel << "    \n";
    kernel << "    // Write result\n";
    kernel << "    if (row < M && col < N) {\n";
    kernel << "        C[row * N + col] = sum;\n";
    kernel << "    }\n";
    kernel << "}\n";

    return kernel.str();
}

// GPUMatmulOptimizer implementation
GPUMatmulOptimizer::BlockConfig GPUMatmulOptimizer::getOptimalConfig(int M, int N, int K) {
    BlockConfig config;

    // GPU推荐的分块大小
    config.blockM = DEFAULT_BLOCK_M;
    config.blockN = DEFAULT_BLOCK_N;
    config.blockK = DEFAULT_BLOCK_K;

    // 检查是否可以使用Tensor Core（需要较大的矩阵且维度对齐）
    bool dimAligned = (M % TENSOR_CORE_ALIGNMENT == 0) &&
                      (N % TENSOR_CORE_ALIGNMENT == 0) &&
                      (K % TENSOR_CORE_ALIGNMENT == 0);
    bool dimLargeEnough = (M >= TENSOR_CORE_MIN_DIM) &&
                         (N >= TENSOR_CORE_MIN_DIM) &&
                         (K >= TENSOR_CORE_MIN_DIM);

    config.useTensorCore = dimAligned && dimLargeEnough;

    // 线程块大小（32x32 = 1024 threads，适合大多数GPU）
    config.threadsPerBlock = config.blockM * config.blockN;

    // TODO: Get actual SM count from CUDA runtime
    config.smCount = 80; // 假设80个SM（例如RTX 3080）

    std::cout << "[GPU] Optimal config for " << M << "x" << N << "x" << K
              << ": blocks(" << config.blockM << "," << config.blockN << "," << config.blockK
              << "), tensor_core=" << config.useTensorCore
              << ", threads=" << config.threadsPerBlock << std::endl;

    return config;
}

std::string GPUMatmulOptimizer::generateOptimizedKernel(const BlockConfig& config) {
    return GPUBackend::generateCUDAMatmulKernel(
        config.blockM,
        config.blockN,
        config.blockK,
        config.useTensorCore
    );
}

} // namespace matrix
