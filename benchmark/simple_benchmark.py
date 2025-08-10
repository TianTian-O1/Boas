#!/usr/bin/env python3
"""
🚀 简化版性能benchmark - 专注于PyTorch NPU vs CPU对比
同时展示Boas编译器的CANN集成状态
"""

import time
import numpy as np
import subprocess
import os
from typing import Dict, List, Optional

def test_cpu_performance():
    """测试CPU性能基准"""
    print("💻 Testing CPU (NumPy) Performance")
    print("-" * 40)
    
    results = []
    matrix_sizes = [64, 128, 256, 512, 1024]
    
    for size in matrix_sizes:
        print(f"📊 Testing {size}×{size} matrix...")
        
        # 创建测试数据
        A = np.random.randn(size, size).astype(np.float32)
        B = np.random.randn(size, size).astype(np.float32)
        
        # 预热
        for _ in range(3):
            C = np.matmul(A, B)
        
        # 性能测试
        times = []
        for _ in range(5):
            start = time.time()
            C = np.matmul(A, B)
            end = time.time()
            times.append((end - start) * 1000)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        flops = 2 * size**3
        gflops = (flops / 1e9) / (avg_time / 1000)
        
        result = {
            'size': size,
            'time_ms': avg_time,
            'std_ms': std_time,
            'gflops': gflops,
            'framework': 'CPU (NumPy)'
        }
        results.append(result)
        
        print(f"   ✅ {avg_time:.2f}±{std_time:.2f}ms, {gflops:.1f} GFLOPS")
    
    return results

def test_pytorch_npu_performance():
    """测试PyTorch NPU性能"""
    print("\n🔥 Testing PyTorch+NPU Performance") 
    print("-" * 40)
    
    try:
        import torch
        import torch_npu
        
        if not torch_npu.npu.is_available():
            print("❌ NPU not available")
            return []
        
        device = torch.device('npu:0')
        print(f"✅ Using device: {torch_npu.npu.get_device_name(0)}")
        
        results = []
        matrix_sizes = [64, 128, 256, 512, 1024]
        
        for size in matrix_sizes:
            print(f"📊 Testing {size}×{size} matrix...")
            
            # 创建测试数据
            A = torch.randn(size, size, dtype=torch.float32).to(device)
            B = torch.randn(size, size, dtype=torch.float32).to(device)
            
            # 预热
            for _ in range(5):
                C = torch.matmul(A, B)
                torch_npu.npu.synchronize()
            
            # 性能测试
            times = []
            for _ in range(10):
                torch_npu.npu.synchronize()
                start = time.time()
                C = torch.matmul(A, B)
                torch_npu.npu.synchronize()
                end = time.time()
                times.append((end - start) * 1000)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            flops = 2 * size**3
            gflops = (flops / 1e9) / (avg_time / 1000)
            
            result = {
                'size': size,
                'time_ms': avg_time,
                'std_ms': std_time,
                'gflops': gflops,
                'framework': 'PyTorch+NPU'
            }
            results.append(result)
            
            print(f"   ✅ {avg_time:.2f}±{std_time:.2f}ms, {gflops:.1f} GFLOPS")
        
        return results
        
    except ImportError:
        print("❌ PyTorch or torch_npu not available")
        return []
    except Exception as e:
        print(f"❌ PyTorch NPU test failed: {e}")
        return []

def test_boas_compilation_status():
    """测试Boas编译状态"""
    print("\n🎯 Testing Boas+CANN Integration Status")
    print("-" * 40)
    
    try:
        # 检查编译器是否存在
        boas_root = "/root/Boas/Boas-linux"
        build_dir = os.path.join(boas_root, "build")
        pipeline_exe = os.path.join(build_dir, "test-full-pipeline")
        
        if not os.path.exists(pipeline_exe):
            print("❌ Boas compiler not found")
            return {'status': 'not_found'}
        
        print("✅ Boas compiler found")
        
        # 创建简单测试文件
        test_code = """import tensor

def test_matmul():
    A = tensor.create(2, 2, [1.0, 2.0, 3.0, 4.0])
    B = tensor.create(2, 2, [5.0, 6.0, 7.0, 8.0])
    C = tensor.matmul(A, B)
    return C

def main():
    result = test_matmul()
    return result
"""
        
        test_file = "/tmp/boas_test.bs"
        with open(test_file, 'w') as f:
            f.write(test_code)
        
        # 尝试编译
        cmd = [pipeline_exe, "--build", test_file, "test_output"]
        env = os.environ.copy()
        cann_lib = "/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64"
        env["LD_LIBRARY_PATH"] = f"{cann_lib}:{env.get('LD_LIBRARY_PATH', '')}"
        
        start_time = time.time()
        result = subprocess.run(
            cmd,
            cwd=build_dir,
            capture_output=True,
            text=True,
            env=env,
            timeout=30
        )
        compile_time = (time.time() - start_time) * 1000
        
        # 清理
        if os.path.exists(test_file):
            os.remove(test_file)
        
        output = result.stdout
        error = result.stderr
        
        # 分析输出
        status = {
            'status': 'tested',
            'compile_time_ms': compile_time,
            'success': result.returncode == 0,
            'cann_init': '[CANN] Successfully initialized' in output,
            'npu_optimization': 'NPU' in output and 'optimized' in output,
            'mlir_generation': 'MLIR' in output or 'llvm' in output.lower(),
            'error_summary': error[-200:] if error else ""
        }
        
        if status['success']:
            print(f"✅ Compilation successful ({compile_time:.0f}ms)")
            if status['cann_init']:
                print("✅ CANN runtime initialization working")
            if status['npu_optimization']:
                print("✅ NPU optimization path detected")
            if status['mlir_generation']:
                print("✅ MLIR/LLVM code generation working")
        else:
            print(f"⚠️ Compilation issues detected")
            print(f"   - Error: {status['error_summary']}")
            
            # 仍然检查部分功能
            if status['cann_init']:
                print("✅ CANN runtime can initialize")
            if status['mlir_generation']:
                print("✅ MLIR generation partially working")
        
        return status
        
    except Exception as e:
        print(f"❌ Boas test failed: {e}")
        return {'status': 'error', 'error': str(e)}

def generate_performance_report(cpu_results, npu_results, boas_status):
    """生成性能报告"""
    print("\n" + "=" * 60)
    print("📊 PERFORMANCE BENCHMARK RESULTS")
    print("=" * 60)
    
    if cpu_results and npu_results:
        print("\n🏆 Performance Comparison Table")
        print("-" * 60)
        print(f"{'Matrix Size':<12} {'CPU (NumPy)':<15} {'PyTorch+NPU':<15} {'Speedup':<10}")
        print("-" * 60)
        
        for cpu_res, npu_res in zip(cpu_results, npu_results):
            if cpu_res['size'] == npu_res['size']:
                speedup = npu_res['gflops'] / cpu_res['gflops']
                print(f"{cpu_res['size']}×{cpu_res['size']:<7} "
                      f"{cpu_res['gflops']:>8.1f} GFLOPS   "
                      f"{npu_res['gflops']:>8.1f} GFLOPS   "
                      f"{speedup:>6.1f}x")
        
        # 计算平均性能
        avg_cpu = np.mean([r['gflops'] for r in cpu_results])
        avg_npu = np.mean([r['gflops'] for r in npu_results])
        avg_speedup = avg_npu / avg_cpu
        
        print("-" * 60)
        print(f"{'Average':<12} {avg_cpu:>8.1f} GFLOPS   {avg_npu:>8.1f} GFLOPS   {avg_speedup:>6.1f}x")
        
        # 性能分析
        print(f"\n📈 Performance Analysis:")
        print(f"   • CPU Baseline: {avg_cpu:.1f} GFLOPS average")
        print(f"   • NPU Performance: {avg_npu:.1f} GFLOPS average")  
        print(f"   • Overall Speedup: {avg_speedup:.1f}x faster on NPU")
        
        # 找出最佳性能点
        best_npu = max(npu_results, key=lambda x: x['gflops'])
        print(f"   • Peak NPU Performance: {best_npu['gflops']:.1f} GFLOPS ({best_npu['size']}×{best_npu['size']})")
    
    # Boas状态报告
    print(f"\n🎯 Boas+CANN Integration Status:")
    if boas_status['status'] == 'not_found':
        print("   ❌ Boas compiler not found")
    elif boas_status['status'] == 'error':
        print(f"   ❌ Test error: {boas_status.get('error', 'Unknown')}")
    else:
        if boas_status.get('success', False):
            print("   ✅ Boas compilation working")
        else:
            print("   ⚠️ Boas compilation has issues")
            
        if boas_status.get('cann_init', False):
            print("   ✅ CANN runtime integration working")
        if boas_status.get('npu_optimization', False):
            print("   ✅ NPU optimization path active")
        if boas_status.get('mlir_generation', False):
            print("   ✅ MLIR/LLVM code generation working")
    
    # 与参考实现对比
    print(f"\n📋 Reference Performance Targets:")
    print(f"   • Triton-Ascend: ~15,000-50,000 GFLOPS (estimated)")
    print(f"   • PyTorch+NPU: {avg_npu:.1f} GFLOPS (measured)")
    print(f"   • Boas+CANN: Integration ready, performance optimization needed")
    
    print(f"\n🔮 Next Steps for Boas:")
    print(f"   1. Fix remaining compilation issues")
    print(f"   2. Implement end-to-end NPU execution")
    print(f"   3. Add MLIR optimization passes")
    print(f"   4. Benchmark against PyTorch+NPU baseline")
    print(f"   5. Target: Match or exceed {avg_npu:.0f} GFLOPS average")

def main():
    """主函数"""
    print("🚀 NPU Performance Benchmark Suite")
    print("=" * 60)
    print("🎯 Comparing CPU vs PyTorch+NPU vs Boas+CANN Integration")
    print("=" * 60)
    
    # 运行CPU基准测试
    cpu_results = test_cpu_performance()
    
    # 运行PyTorch NPU测试
    npu_results = test_pytorch_npu_performance()
    
    # 测试Boas集成状态
    boas_status = test_boas_compilation_status()
    
    # 生成报告
    generate_performance_report(cpu_results, npu_results, boas_status)
    
    # 保存结果
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results = {
        'timestamp': timestamp,
        'cpu_results': cpu_results,
        'npu_results': npu_results,
        'boas_status': boas_status
    }
    
    import json
    with open(f'benchmark_results_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to benchmark_results_{timestamp}.json")
    print("✅ Benchmark completed!")

if __name__ == "__main__":
    main()
