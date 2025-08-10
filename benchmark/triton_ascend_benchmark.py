#!/usr/bin/env python3
"""
⚡ Triton-Ascend性能测试集成
参考 https://gitee.com/ascend/triton-ascend.git 实现
"""

import os
import sys
import time
import subprocess
import tempfile
from typing import Optional, Dict, Any

# 尝试导入triton相关模块
try:
    import triton
    import triton.language as tl
    from triton.runtime.driver import get_current_target
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

class TritonAscendMatmul:
    """Triton-Ascend矩阵乘法实现"""
    
    def __init__(self):
        self.kernel_cache = {}
    
    def get_triton_matmul_kernel(self) -> str:
        """获取Triton矩阵乘法kernel代码"""
        return """
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    # 指针参数
    a_ptr, b_ptr, c_ptr,
    # 矩阵维度
    M, N, K,
    # stride参数
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # 块大小
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    # 激活函数
    ACTIVATION: tl.constexpr
):
    # 获取程序ID
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    # 计算当前块的行列索引
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    # 创建偏移量
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # 计算指针偏移
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    # 累加器
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # 主循环
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # 加载数据
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        
        # 矩阵乘法
        accumulator += tl.dot(a, b)
        
        # 更新指针
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    # 输出偏移量
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    
    # 存储结果
    tl.store(c_ptrs, accumulator, mask=c_mask)

def triton_matmul_ascend(a, b):
    \"\"\"
    使用Triton在Ascend NPU上执行矩阵乘法
    \"\"\"
    # 检查输入
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.device == b.device, "Tensors must be on same device"
    
    M, K = a.shape
    K, N = b.shape
    
    # 创建输出tensor
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # 块大小优化 - 针对Ascend NPU
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    
    # 计算grid大小
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),
    )
    
    # 启动kernel
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        ACTIVATION="none"
    )
    
    return c
"""
    
    def create_triton_test_file(self, matrix_size: int) -> str:
        """创建Triton测试文件"""
        code = f"""
{self.get_triton_matmul_kernel()}

import torch
import time
import numpy as np

def benchmark_triton_matmul():
    device = torch.device('npu:0' if torch.npu.is_available() else 'cpu')
    
    # 创建测试数据
    M, N, K = {matrix_size}, {matrix_size}, {matrix_size}
    a = torch.randn((M, K), device=device, dtype=torch.float32)
    b = torch.randn((K, N), device=device, dtype=torch.float32)
    
    # 预热
    for _ in range(3):
        c = triton_matmul_ascend(a, b)
        torch.npu.synchronize()
    
    # 性能测试
    times = []
    for _ in range(10):
        start_time = time.time()
        c = triton_matmul_ascend(a, b)
        torch.npu.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = np.mean(times) * 1000  # 转换为ms
    flops = 2 * M * N * K
    gflops = (flops / 1e9) / (avg_time / 1000)
    
    print(f"Triton-Ascend {{matrix_size}}x{{matrix_size}}: {{avg_time:.2f}}ms, {{gflops:.1f}} GFLOPS")
    
    return avg_time, gflops

if __name__ == "__main__":
    benchmark_triton_matmul()
"""
        return code
    
    def run_triton_benchmark(self, matrix_size: int) -> Dict[str, Any]:
        """运行Triton-Ascend benchmark"""
        if not HAS_TRITON:
            return {
                'success': False,
                'error': 'Triton not available',
                'execution_time_ms': 0,
                'gflops': 0
            }
        
        try:
            # 创建临时测试文件
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(self.create_triton_test_file(matrix_size))
                test_file = f.name
            
            # 运行测试
            result = subprocess.run(
                [sys.executable, test_file],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            # 清理临时文件
            os.unlink(test_file)
            
            if result.returncode == 0:
                # 解析输出
                output = result.stdout.strip()
                if "GFLOPS" in output:
                    # 提取性能数据
                    parts = output.split()
                    time_ms = float(parts[2].replace('ms,', ''))
                    gflops = float(parts[3])
                    
                    return {
                        'success': True,
                        'execution_time_ms': time_ms,
                        'gflops': gflops,
                        'output': output
                    }
            
            return {
                'success': False,
                'error': f"Execution failed: {result.stderr}",
                'execution_time_ms': 0,
                'gflops': 0
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'execution_time_ms': 0,
                'gflops': 0
            }

class TritonAscendSetup:
    """Triton-Ascend环境设置"""
    
    @staticmethod
    def check_triton_ascend_installation() -> bool:
        """检查triton-ascend是否正确安装"""
        try:
            # 检查是否可以导入triton
            import triton
            
            # 检查是否支持NPU后端
            targets = triton.runtime.driver.get_active_targets()
            ascend_supported = any('ascend' in str(target).lower() for target in targets)
            
            return ascend_supported
        except:
            return False
    
    @staticmethod
    def install_triton_ascend():
        """安装triton-ascend"""
        print("🔧 Installing triton-ascend...")
        
        # 克隆triton-ascend仓库
        clone_cmd = [
            "git", "clone", 
            "https://gitee.com/ascend/triton-ascend.git",
            "/tmp/triton-ascend"
        ]
        
        try:
            subprocess.run(clone_cmd, check=True)
            
            # 安装
            install_cmd = [
                "pip", "install", "-e", "/tmp/triton-ascend"
            ]
            subprocess.run(install_cmd, check=True)
            
            print("✅ triton-ascend installed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install triton-ascend: {e}")
            return False

def main():
    """测试Triton-Ascend性能"""
    print("⚡ Triton-Ascend Benchmark")
    print("=" * 50)
    
    # 检查安装
    if not TritonAscendSetup.check_triton_ascend_installation():
        print("❌ Triton-Ascend not properly installed")
        print("🔧 Attempting to install...")
        if not TritonAscendSetup.install_triton_ascend():
            print("❌ Installation failed")
            sys.exit(1)
    
    # 运行benchmark
    triton_bench = TritonAscendMatmul()
    
    matrix_sizes = [64, 128, 256, 512, 1024]
    
    for size in matrix_sizes:
        print(f"\n📊 Testing {size}×{size} matrix...")
        result = triton_bench.run_triton_benchmark(size)
        
        if result['success']:
            print(f"✅ Triton-Ascend: {result['execution_time_ms']:.2f}ms, "
                  f"{result['gflops']:.1f} GFLOPS")
        else:
            print(f"❌ Triton-Ascend failed: {result['error']}")

if __name__ == "__main__":
    main()
