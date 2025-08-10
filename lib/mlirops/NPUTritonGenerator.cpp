#include "mlirops/NPUBackend.h"
#include <fstream>
#include <sstream>
#include <filesystem>

namespace matrix {

/// NPU Triton Kernel Generator
/// 生成针对昇腾NPU优化的Triton kernel代码
class NPUTritonGenerator {
public:
    /// 生成完整的NPU优化矩阵乘法kernel
    static std::string generateFullMatmulKernel(const NPUMatmulOptimizer::BlockConfig& config) {
        std::stringstream kernel;
        
        // 生成Python导入和设备设置
        kernel << generatePythonHeader();
        kernel << "\n";
        
        // 生成设备属性获取函数
        kernel << generateDevicePropertiesFunc();
        kernel << "\n";
        
        // 生成主要的Triton kernel
        kernel << generateTritonKernel(config);
        kernel << "\n";
        
        // 生成wrapper函数
        kernel << generateWrapperFunction(config);
        kernel << "\n";
        
        // 生成测试和验证代码
        kernel << generateTestCode();
        
        return kernel.str();
    }
    
    /// 将生成的kernel保存到文件
    static bool saveKernelToFile(const std::string& kernel, const std::string& filename) {
        try {
            std::ofstream file(filename);
            if (!file.is_open()) {
                std::cerr << "[NPU] Failed to create kernel file: " << filename << std::endl;
                return false;
            }
            
            file << kernel;
            file.close();
            
            std::cout << "[NPU] Saved Triton kernel to: " << filename << std::endl;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "[NPU] Error saving kernel: " << e.what() << std::endl;
            return false;
        }
    }

private:
    /// 生成Python头部和导入
    static std::string generatePythonHeader() {
        return R"(#!/usr/bin/env python3
"""
Boas Language NPU-optimized Matrix Multiplication Kernel
Generated automatically for Ascend NPU devices
"""

import torch
import torch_npu
import triton
import triton.language as tl
import triton.runtime.driver as driver

# NPU设备配置
DEV = "npu"
torch.npu.set_device(0)  # 使用第一个NPU设备
)";
    }
    
    /// 生成设备属性获取函数
    static std::string generateDevicePropertiesFunc() {
        return R"(def get_npu_properties():
    """获取NPU设备属性"""
    device = torch.npu.current_device()
    return driver.active.utils.get_device_properties(device)

def get_optimal_num_cores():
    """获取最优核心数量"""
    props = get_npu_properties()
    return props.get("num_aicore", 20)  # 默认20个AI Core
)";
    }
    
    /// 生成主要的Triton kernel
    static std::string generateTritonKernel(const NPUMatmulOptimizer::BlockConfig& config) {
        std::stringstream kernel;
        
        kernel << "@triton.jit\n";
        kernel << "def boas_npu_matmul_kernel(\n";
        kernel << "    mat_a, mat_b, mat_c,\n";
        kernel << "    M: tl.constexpr,\n";
        kernel << "    N: tl.constexpr,\n";
        kernel << "    K: tl.constexpr,\n";
        kernel << "    BLOCK_M: tl.constexpr,\n";
        kernel << "    BLOCK_N: tl.constexpr,\n";
        kernel << "    BLOCK_K: tl.constexpr,\n";
        kernel << "    NUM_BLOCKS: tl.constexpr,\n";
        kernel << "    NUM_BLOCKS_N: tl.constexpr,\n";
        kernel << "    NUM_BLOCKS_M: tl.constexpr,\n";
        kernel << "    num_cores: tl.constexpr,\n";
        kernel << "):\n";
        kernel << "    \"\"\"Boas NPU优化矩阵乘法kernel\"\"\"\n";
        kernel << "    pid = tl.program_id(axis=0)\n";
        kernel << "    task_m_idx = 0\n";
        kernel << "    task_n_idx = 0\n\n";
        
        if (config.useDiagonalTiling) {
            kernel << generateDiagonalTilingLogic();
        } else {
            kernel << generateSequentialTilingLogic();
        }
        
        return kernel.str();
    }
    
    /// 生成对角线分核逻辑
    static std::string generateDiagonalTilingLogic() {
        return R"(    # NPU优化：8x8对角线分核策略
    # 当矩阵在M和N方向均超过8块时启用对角线分核可以明显减小Bank冲突
    BLOCK_THRESHOLD = 8
    if NUM_BLOCKS_M >= BLOCK_THRESHOLD and NUM_BLOCKS_N >= BLOCK_THRESHOLD:
        for block_idx in range(pid, NUM_BLOCKS, num_cores):
            # 8x8对角线分核代码实现
            curThresholdM = BLOCK_THRESHOLD if block_idx < (NUM_BLOCKS_M // BLOCK_THRESHOLD * BLOCK_THRESHOLD) * NUM_BLOCKS_N else NUM_BLOCKS_M % BLOCK_THRESHOLD
            curThresholdM_thresholdN = curThresholdM * BLOCK_THRESHOLD
            curThresholdN = BLOCK_THRESHOLD if block_idx % (NUM_BLOCKS_N * BLOCK_THRESHOLD) < (curThresholdM * NUM_BLOCKS_N) // curThresholdM_thresholdN * curThresholdM_thresholdN else NUM_BLOCKS_N % BLOCK_THRESHOLD
            
            localRelativeBlock = block_idx % (BLOCK_THRESHOLD * NUM_BLOCKS_N) % (BLOCK_THRESHOLD * curThresholdM)
            task_m_idx = localRelativeBlock % curThresholdM + block_idx // (BLOCK_THRESHOLD * NUM_BLOCKS_N) * BLOCK_THRESHOLD
            
            # 求最小公倍数，方便求基本块的坐标
            x, y = curThresholdM, curThresholdN if curThresholdM > curThresholdN else curThresholdN, curThresholdM
            while y != 0:
                x, y = y, x % y
            lcm = curThresholdM * curThresholdN // x
            task_n_idx = (localRelativeBlock + (localRelativeBlock // lcm)) % curThresholdN + block_idx % (BLOCK_THRESHOLD * NUM_BLOCKS_N) // curThresholdM_thresholdN * BLOCK_THRESHOLD
            
            # 执行矩阵乘法计算
            _perform_matmul_computation(mat_a, mat_b, mat_c, task_m_idx, task_n_idx, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K)
    else:
        # 当块数量较少时，使用传统顺序分核
        for block_idx in range(pid, NUM_BLOCKS, num_cores):
            task_m_idx = block_idx // NUM_BLOCKS_N
            task_n_idx = block_idx % NUM_BLOCKS_N
            _perform_matmul_computation(mat_a, mat_b, mat_c, task_m_idx, task_n_idx, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K)

@triton.jit
def _perform_matmul_computation(mat_a, mat_b, mat_c, task_m_idx, task_n_idx, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K):
    """执行实际的矩阵乘法计算"""
    m_start = task_m_idx * BLOCK_M
    n_start = task_n_idx * BLOCK_N
    
    # 初始化累加器为FP32以提高精度
    mat_c_block = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # K维度循环
    for k_start in range(0, K, BLOCK_K):
        # 加载A矩阵块
        mat_a_offset = ((m_start + tl.arange(0, BLOCK_M)) * K)[:, None] + (k_start + tl.arange(0, BLOCK_K))[None, :]
        mat_a_mask = ((m_start + tl.arange(0, BLOCK_M)) < M)[:, None] & ((k_start + tl.arange(0, BLOCK_K)) < K)[None, :]
        mat_a_block = tl.load(mat_a + mat_a_offset, mask=mat_a_mask, other=0.0)
        
        # 加载B矩阵块
        mat_b_offset = ((k_start + tl.arange(0, BLOCK_K)) * N)[:, None] + (n_start + tl.arange(0, BLOCK_N))[None, :]
        mat_b_mask = ((k_start + tl.arange(0, BLOCK_K)) < K)[:, None] & ((n_start + tl.arange(0, BLOCK_N)) < N)[None, :]
        mat_b_block = tl.load(mat_b + mat_b_offset, mask=mat_b_mask, other=0.0)
        
        # 执行矩阵乘法累积
        mat_c_block = tl.dot(mat_a_block, mat_b_block, mat_c_block)
    
    # 存储结果（转换为bfloat16以节省内存带宽）
    mat_c_offset = ((m_start + tl.arange(0, BLOCK_M)) * N)[:, None] + (n_start + tl.arange(0, BLOCK_N))[None, :]
    mat_c_mask = ((m_start + tl.arange(0, BLOCK_M)) < M)[:, None] & ((n_start + tl.arange(0, BLOCK_N)) < N)[None, :]
    tl.store(mat_c + mat_c_offset, mat_c_block.to(tl.bfloat16), mask=mat_c_mask)
)";
    }
    
    /// 生成顺序分核逻辑
    static std::string generateSequentialTilingLogic() {
        return R"(    # 传统顺序分核策略
    for block_idx in range(pid, NUM_BLOCKS, num_cores):
        task_m_idx = block_idx // NUM_BLOCKS_N
        task_n_idx = block_idx % NUM_BLOCKS_N
        
        m_start = task_m_idx * BLOCK_M
        n_start = task_n_idx * BLOCK_N
        
        mat_c_block = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        for k_start in range(0, K, BLOCK_K):
            mat_a_offset = ((m_start + tl.arange(0, BLOCK_M)) * K)[:, None] + (k_start + tl.arange(0, BLOCK_K))[None, :]
            mat_a_mask = ((m_start + tl.arange(0, BLOCK_M)) < M)[:, None] & ((k_start + tl.arange(0, BLOCK_K)) < K)[None, :]
            mat_a_block = tl.load(mat_a + mat_a_offset, mask=mat_a_mask, other=0.0)

            mat_b_offset = ((k_start + tl.arange(0, BLOCK_K)) * N)[:, None] + (n_start + tl.arange(0, BLOCK_N))[None, :]
            mat_b_mask = ((k_start + tl.arange(0, BLOCK_K)) < K)[:, None] & ((n_start + tl.arange(0, BLOCK_N)) < N)[None, :]
            mat_b_block = tl.load(mat_b + mat_b_offset, mask=mat_b_mask, other=0.0)
            
            mat_c_block = tl.dot(mat_a_block, mat_b_block, mat_c_block)
        
        mat_c_offset = ((m_start + tl.arange(0, BLOCK_M)) * N)[:, None] + (n_start + tl.arange(0, BLOCK_N))[None, :]
        mat_c_mask = ((m_start + tl.arange(0, BLOCK_M)) < M)[:, None] & ((n_start + tl.arange(0, BLOCK_N)) < N)[None, :]
        tl.store(mat_c + mat_c_offset, mat_c_block.to(tl.bfloat16), mask=mat_c_mask)
)";
    }
    
    /// 生成wrapper函数
    static std::string generateWrapperFunction(const NPUMatmulOptimizer::BlockConfig& config) {
        std::stringstream wrapper;
        
        wrapper << "def boas_npu_matmul(mat_a, mat_b):\n";
        wrapper << "    \"\"\"Boas NPU矩阵乘法wrapper函数\"\"\"\n";
        wrapper << "    m = mat_a.shape[0]\n";
        wrapper << "    k = mat_a.shape[1]\n";
        wrapper << "    n = mat_b.shape[1]\n";
        wrapper << "    \n";
        wrapper << "    # 创建输出张量\n";
        wrapper << "    mat_c = torch.empty(m, n, dtype=mat_a.dtype, device=mat_a.device)\n";
        wrapper << "    \n";
        wrapper << "    # NPU优化块配置\n";
        wrapper << "    BLOCK_M = " << config.blockM << "\n";
        wrapper << "    BLOCK_N = " << config.blockN << "\n";
        wrapper << "    BLOCK_K = " << config.blockK << "\n";
        wrapper << "    \n";
        wrapper << "    num_cores = get_optimal_num_cores()\n";
        wrapper << "    NUM_BLOCKS_N = triton.cdiv(n, BLOCK_N)\n";
        wrapper << "    NUM_BLOCKS_M = triton.cdiv(m, BLOCK_M)\n";
        wrapper << "    NUM_BLOCKS = NUM_BLOCKS_M * NUM_BLOCKS_N\n";
        wrapper << "    \n";
        wrapper << "    # 启动kernel\n";
        wrapper << "    grid = (num_cores,)\n";
        wrapper << "    boas_npu_matmul_kernel[grid](\n";
        wrapper << "        mat_a, mat_b, mat_c,\n";
        wrapper << "        m, n, k,\n";
        wrapper << "        BLOCK_M, BLOCK_N, BLOCK_K,\n";
        wrapper << "        NUM_BLOCKS, NUM_BLOCKS_N, NUM_BLOCKS_M,\n";
        wrapper << "        num_cores\n";
        wrapper << "    )\n";
        wrapper << "    \n";
        wrapper << "    return mat_c\n";
        
        return wrapper.str();
    }
    
    /// 生成测试代码
    static std::string generateTestCode() {
        return R"(
def test_boas_npu_matmul():
    """测试Boas NPU矩阵乘法"""
    print("[Boas NPU] Running matrix multiplication test...")
    
    # 测试配置
    M, K, N = 2048, 1024, 2048
    
    # 创建测试矩阵
    mat_a = torch.randn([M, K], dtype=torch.bfloat16, device=DEV)
    mat_b = torch.randn([K, N], dtype=torch.bfloat16, device=DEV)
    
    # Boas NPU实现
    result_boas = boas_npu_matmul(mat_a, mat_b)
    
    # PyTorch参考实现
    result_torch = torch.matmul(mat_a, mat_b)
    
    # 精度验证
    mask = result_torch.abs() < 1.0
    atol, rtol = 2 ** -6, 2 ** -6
    
    try:
        torch.testing.assert_close(result_boas[mask], result_torch[mask], atol=atol, rtol=0)
        torch.testing.assert_close(result_boas[~mask], result_torch[~mask], atol=0, rtol=rtol)
        print(f"[Boas NPU] Test PASSED for {M}x{K}x{N} matrix multiplication")
        return True
    except Exception as e:
        print(f"[Boas NPU] Test FAILED for {M}x{K}x{N}: {e}")
        return False

if __name__ == "__main__":
    print("Boas Language NPU Matrix Multiplication Kernel")
    print("=" * 50)
    
    # 检查NPU可用性
    if torch.npu.is_available():
        print(f"NPU device count: {torch.npu.device_count()}")
        print(f"Current NPU device: {torch.npu.current_device()}")
        
        # 运行测试
        test_boas_npu_matmul()
    else:
        print("No NPU devices available")
)";
    }
};

/// 为NPUBackend添加Triton kernel生成功能
std::string NPUBackend::generateTritonMatmulKernel(
    int blockM,
    int blockN, 
    int blockK,
    bool enableDiagonalTiling) {
    
    NPUMatmulOptimizer::BlockConfig config;
    config.blockM = blockM;
    config.blockN = blockN;
    config.blockK = blockK;
    config.useDiagonalTiling = enableDiagonalTiling;
    config.numCores = 20; // 默认值
    
    return NPUTritonGenerator::generateFullMatmulKernel(config);
}

} // namespace matrix
