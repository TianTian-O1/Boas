#include "mlirops/NPUBackend.h"
#include "mlirops/CANNRuntime.h"
#include <iostream>
#include <sstream>
#include <cmath>

namespace matrix {

// Static member initialization
bool NPUBackend::initialized_ = false;
int NPUBackend::currentDevice_ = 0;
int NPUBackend::deviceCount_ = 0;

bool NPUBackend::initialize() {
    if (initialized_) return true;
    
    std::cout << "[NPU] Initializing Ascend NPU backend..." << std::endl;
    
    // 使用真正的CANN运行时初始化
    auto& cannRuntime = CANNRuntime::getInstance();
    if (!cannRuntime.initialize()) {
        std::cerr << "[NPU] Failed to initialize CANN runtime: " << cannRuntime.getLastError() << std::endl;
        return false;
    }
    
    deviceCount_ = cannRuntime.getDeviceCount();
    currentDevice_ = cannRuntime.getCurrentDevice();
    initialized_ = true;
    
    std::cout << "[NPU] Successfully initialized with " << deviceCount_ << " device(s)" << std::endl;
    std::cout << "[NPU] Current device: " << cannRuntime.getDeviceProperties(currentDevice_) << std::endl;
    return true;
}

bool NPUBackend::isAvailable() {
    // 检查CANN运行时是否可用
    auto& cannRuntime = CANNRuntime::getInstance();
    if (!cannRuntime.isAvailable()) {
        // 尝试初始化
        if (!initialize()) {
            return false;
        }
    }
    return cannRuntime.isAvailable();
}

int NPUBackend::getDeviceCount() {
    if (!initialized_) {
        initialize();
    }
    return deviceCount_;
}

bool NPUBackend::setDevice(int deviceId) {
    if (deviceId < 0 || deviceId >= deviceCount_) {
        std::cerr << "[NPU] Invalid device ID: " << deviceId << std::endl;
        return false;
    }
    
    currentDevice_ = deviceId;
    std::cout << "[NPU] Set current device to " << deviceId << std::endl;
    return true;
}

std::string NPUBackend::getDeviceProperties() {
    std::stringstream props;
    props << "NPU Device " << currentDevice_ << ": Ascend 910A";
    return props.str();
}

mlir::Value NPUBackend::generateNPUMatmul(
    MLIRGen* generator,
    mlir::Value lhs, 
    mlir::Value rhs,
    mlir::Value M, 
    mlir::Value N, 
    mlir::Value K) {
    
    std::cout << "[NPU] Generating CANN-optimized matrix multiplication" << std::endl;
    
    auto loc = generator->getBuilder()->getUnknownLoc();
    
    // Create result tensor
    auto resultType = mlir::MemRefType::get(
        {mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic},
        generator->getBuilder()->getF64Type()
    );
    
    auto result = generator->getBuilder()->create<mlir::memref::AllocOp>(
        loc, resultType, mlir::ValueRange{M, N}
    );
    
    // 生成NPU优化的linalg.matmul，添加特殊属性标记
    auto matmulOp = generator->getBuilder()->create<mlir::linalg::MatmulOp>(
        loc, 
        mlir::ValueRange{lhs, rhs}, 
        mlir::ValueRange{result}
    );
    
    // 添加NPU优化属性
    auto stringAttr = generator->getBuilder()->getStringAttr("npu_optimized");
    matmulOp->setAttr("boas.backend", stringAttr);
    
    // 添加设备属性
    auto deviceAttr = generator->getBuilder()->getStringAttr("ascend_npu");
    matmulOp->setAttr("boas.device", deviceAttr);
    
    // 添加优化策略属性
    auto strategyAttr = generator->getBuilder()->getStringAttr("cann_matmul");
    matmulOp->setAttr("boas.strategy", strategyAttr);
    
    std::cout << "[NPU] Generated CANN-optimized linalg.matmul with NPU attributes" << std::endl;
    return result;
}

std::string NPUBackend::generateTritonMatmulKernel(
    int blockM,
    int blockN, 
    int blockK,
    bool enableDiagonalTiling) {
    
    std::stringstream kernel;
    
    kernel << "# NPU-optimized Triton matrix multiplication kernel\n";
    kernel << "# Block sizes: M=" << blockM << ", N=" << blockN << ", K=" << blockK << "\n";
    kernel << "# Diagonal tiling: " << (enableDiagonalTiling ? "enabled" : "disabled") << "\n\n";
    
    kernel << "@triton.jit\n";
    kernel << "def npu_matmul_kernel(\n";
    kernel << "    mat_a, mat_b, mat_c,\n";
    kernel << "    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,\n";
    kernel << "    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,\n";
    kernel << "    NUM_BLOCKS: tl.constexpr, NUM_BLOCKS_N: tl.constexpr, NUM_BLOCKS_M: tl.constexpr,\n";
    kernel << "    num_cores: tl.constexpr,\n";
    kernel << "):\n";
    kernel << "    pid = tl.program_id(axis=0)\n";
    
    if (enableDiagonalTiling) {
        kernel << "    # NPU优化：对角线分核策略\n";
        kernel << "    BLOCK_THRESHOLD = 8\n";
        kernel << "    if NUM_BLOCKS_M >= BLOCK_THRESHOLD and NUM_BLOCKS_N >= BLOCK_THRESHOLD:\n";
        kernel << "        # 8x8对角线分核实现\n";
        kernel << "        for block_idx in range(pid, NUM_BLOCKS, num_cores):\n";
        kernel << "            # 对角线分核逻辑实现\n";
        kernel << "            # ... (详细实现)\n";
    } else {
        kernel << "    # 传统顺序分核\n";
        kernel << "    for block_idx in range(pid, NUM_BLOCKS, num_cores):\n";
        kernel << "        task_m_idx = block_idx // NUM_BLOCKS_N\n";
        kernel << "        task_n_idx = block_idx % NUM_BLOCKS_N\n";
    }
    
    kernel << "        # 矩阵乘法计算核心\n";
    kernel << "        m_start = task_m_idx * BLOCK_M\n";
    kernel << "        n_start = task_n_idx * BLOCK_N\n";
    kernel << "        \n";
    kernel << "        mat_c_block = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)\n";
    kernel << "        for k_start in range(0, K, BLOCK_K):\n";
    kernel << "            # 加载矩阵块\n";
    kernel << "            # ... 实现细节\n";
    kernel << "            mat_c_block = tl.dot(mat_a_block, mat_b_block, mat_c_block)\n";
    kernel << "        \n";
    kernel << "        # 存储结果\n";
    kernel << "        tl.store(mat_c + mat_c_offset, mat_c_block.to(tl.bfloat16), mask=mat_c_mask)\n";
    
    return kernel.str();
}

// NPUMatmulOptimizer implementation
NPUMatmulOptimizer::BlockConfig NPUMatmulOptimizer::getOptimalConfig(int M, int N, int K) {
    BlockConfig config;
    
    // NPU芯片更加亲和512B对齐场景的推荐配置
    config.blockM = DEFAULT_BLOCK_M;
    config.blockN = DEFAULT_BLOCK_N;
    config.blockK = DEFAULT_BLOCK_K;
    
    // 计算块数量
    int numBlocksM = (M + config.blockM - 1) / config.blockM;
    int numBlocksN = (N + config.blockN - 1) / config.blockN;
    
    // 当任务量较多时启用对角线分核
    config.useDiagonalTiling = (numBlocksM >= DIAGONAL_TILING_THRESHOLD && 
                               numBlocksN >= DIAGONAL_TILING_THRESHOLD);
    
    // TODO: Get actual number of NPU cores
    config.numCores = 20; // 假设20个AI Core
    
    std::cout << "[NPU] Optimal config for " << M << "x" << N << "x" << K 
              << ": blocks(" << config.blockM << "," << config.blockN << "," << config.blockK 
              << "), diagonal_tiling=" << config.useDiagonalTiling << std::endl;
    
    return config;
}

std::string NPUMatmulOptimizer::generateOptimizedKernel(const BlockConfig& config) {
    return NPUBackend::generateTritonMatmulKernel(
        config.blockM, 
        config.blockN, 
        config.blockK, 
        config.useDiagonalTiling
    );
}

} // namespace matrix
