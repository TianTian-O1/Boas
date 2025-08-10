#!/usr/bin/env python3
"""
🚀 快速性能验证测试
在运行完整benchmark之前，先验证基本功能
"""

import time
import subprocess
import os
import numpy as np

def test_pytorch_npu():
    """测试PyTorch NPU性能"""
    print("🔥 Testing PyTorch NPU performance...")
    
    try:
        import torch
        import torch_npu
        
        if not torch_npu.npu.is_available():
            print("❌ NPU not available for PyTorch")
            return None
        
        device = torch_npu.npu.device(0)
        matrix_size = 512
        
        # 创建测试数据
        A = torch.randn(matrix_size, matrix_size, dtype=torch.float32).to(device)
        B = torch.randn(matrix_size, matrix_size, dtype=torch.float32).to(device)
        
        # 预热
        for _ in range(3):
            C = torch.matmul(A, B)
            torch_npu.npu.synchronize()
        
        # 性能测试
        times = []
        for _ in range(10):
            start = time.time()
            C = torch.matmul(A, B)
            torch_npu.npu.synchronize()
            end = time.time()
            times.append((end - start) * 1000)
        
        avg_time = np.mean(times)
        flops = 2 * matrix_size**3
        gflops = (flops / 1e9) / (avg_time / 1000)
        
        print(f"✅ PyTorch+NPU ({matrix_size}×{matrix_size}): {avg_time:.2f}ms, {gflops:.1f} GFLOPS")
        return {'time_ms': avg_time, 'gflops': gflops}
        
    except ImportError:
        print("❌ PyTorch NPU not available")
        return None
    except Exception as e:
        print(f"❌ PyTorch NPU test failed: {e}")
        return None

def test_cpu_baseline():
    """测试CPU基准性能"""
    print("💻 Testing CPU baseline performance...")
    
    try:
        matrix_size = 512
        
        # 创建测试数据
        A = np.random.randn(matrix_size, matrix_size).astype(np.float32)
        B = np.random.randn(matrix_size, matrix_size).astype(np.float32)
        
        # 预热
        for _ in range(3):
            C = np.matmul(A, B)
        
        # 性能测试
        times = []
        for _ in range(5):  # CPU测试次数少一些
            start = time.time()
            C = np.matmul(A, B)
            end = time.time()
            times.append((end - start) * 1000)
        
        avg_time = np.mean(times)
        flops = 2 * matrix_size**3
        gflops = (flops / 1e9) / (avg_time / 1000)
        
        print(f"✅ CPU NumPy ({matrix_size}×{matrix_size}): {avg_time:.2f}ms, {gflops:.1f} GFLOPS")
        return {'time_ms': avg_time, 'gflops': gflops}
        
    except Exception as e:
        print(f"❌ CPU test failed: {e}")
        return None

def test_boas_compilation():
    """测试Boas编译和CANN集成"""
    print("🎯 Testing Boas compilation and CANN integration...")
    
    try:
        # 创建简单的Boas测试文件
        boas_code = """import tensor

def simple_matmul():
    A = tensor.create(4, 4, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0])
    B = tensor.create(4, 4, [2.0, 0.0, 1.0, 3.0, 1.0, 2.0, 0.0, 1.0, 3.0, 1.0, 2.0, 0.0, 0.0, 3.0, 1.0, 2.0])
    C = tensor.matmul(A, B)
    return C

def main():
    result = simple_matmul()
    return result
"""
        
        # 写入临时文件
        test_file = "/tmp/quick_boas_test.bs"
        with open(test_file, 'w') as f:
            f.write(boas_code)
        
        # 编译测试
        boas_root = "/root/Boas/Boas-linux"
        build_dir = os.path.join(boas_root, "build")
        pipeline_exe = os.path.join(build_dir, "test-full-pipeline")
        
        if not os.path.exists(pipeline_exe):
            print(f"❌ Boas compiler not found: {pipeline_exe}")
            return None
        
        start_time = time.time()
        
        cmd = [pipeline_exe, "--build", test_file, "quick_test_output"]
        env = os.environ.copy()
        # 设置完整的CANN库路径
        cann_lib_path = "/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64"
        system_lib_path = "/usr/lib/aarch64-linux-gnu"
        old_ld_path = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{cann_lib_path}:{system_lib_path}:{old_ld_path}"
        
        result = subprocess.run(
            cmd,
            cwd=build_dir,
            capture_output=True,
            text=True,
            env=env
        )
        
        compile_time = (time.time() - start_time) * 1000
        
        # 清理临时文件
        if os.path.exists(test_file):
            os.remove(test_file)
        
        if result.returncode == 0:
            # 检查关键输出
            output = result.stdout
            cann_init = "[CANN] Successfully initialized" in output
            npu_detected = "[NPU] Generating CANN-optimized" in output or "NPU-optimized" in output
            mlir_generated = "Generated" in output and "LLVM" in output
            
            print(f"✅ Boas compilation successful:")
            print(f"   - Compile time: {compile_time:.2f}ms")
            print(f"   - CANN initialized: {'✅' if cann_init else '❌'}")
            print(f"   - NPU optimization: {'✅' if npu_detected else '❌'}")
            print(f"   - MLIR/LLVM generated: {'✅' if mlir_generated else '❌'}")
            
            return {
                'compile_time_ms': compile_time,
                'cann_init': cann_init,
                'npu_optimized': npu_detected,
                'mlir_generated': mlir_generated
            }
        else:
            print(f"❌ Boas compilation failed:")
            print(f"   Error: {result.stderr[-200:]}")
            return None
            
    except Exception as e:
        print(f"❌ Boas test failed: {e}")
        return None

def main():
    """主函数"""
    print("🚀 Quick Performance Validation Test")
    print("=" * 60)
    print("🎯 Testing basic functionality before full benchmark")
    print("=" * 60)
    
    results = {}
    
    # 1. 测试Boas编译
    print("\n1️⃣ BOAS COMPILATION TEST")
    print("-" * 30)
    boas_result = test_boas_compilation()
    results['boas'] = boas_result
    
    # 2. 测试PyTorch NPU
    print("\n2️⃣ PYTORCH NPU TEST")
    print("-" * 30)
    pytorch_result = test_pytorch_npu()
    results['pytorch_npu'] = pytorch_result
    
    # 3. 测试CPU基准
    print("\n3️⃣ CPU BASELINE TEST")
    print("-" * 30)
    cpu_result = test_cpu_baseline()
    results['cpu'] = cpu_result
    
    # 总结
    print("\n" + "=" * 60)
    print("📊 QUICK TEST SUMMARY")
    print("=" * 60)
    
    if results['boas']:
        print("✅ Boas+CANN: Compilation and integration working")
        if results['boas']['cann_init']:
            print("   🔥 CANN runtime successfully initialized")
        if results['boas']['npu_optimized']:
            print("   ⚡ NPU optimization path activated")
    else:
        print("❌ Boas+CANN: Issues detected")
    
    if results['pytorch_npu']:
        print(f"✅ PyTorch+NPU: {results['pytorch_npu']['gflops']:.1f} GFLOPS baseline")
    else:
        print("❌ PyTorch+NPU: Not available")
    
    if results['cpu']:
        print(f"✅ CPU Baseline: {results['cpu']['gflops']:.1f} GFLOPS reference")
    else:
        print("❌ CPU Baseline: Issues detected")
    
    # 性能对比预览
    if results['pytorch_npu'] and results['cpu']:
        speedup = results['pytorch_npu']['gflops'] / results['cpu']['gflops']
        print(f"\n🚀 NPU vs CPU speedup: {speedup:.1f}x")
    
    print("\n" + "=" * 60)
    
    # 建议下一步
    all_working = all(r is not None for r in results.values())
    
    if all_working:
        print("🎉 All basic tests passed! Ready for full benchmark.")
        print("\n🚀 Run full benchmark with:")
        print("   cd /root/Boas/Boas-linux/benchmark")
        print("   python3 run_all_benchmarks.py")
    else:
        print("⚠️ Some tests failed. Please check the issues above.")
        print("\n🔧 Troubleshooting:")
        if not results['boas']:
            print("   - Check Boas compilation: cd /root/Boas/Boas-linux/build && make")
        if not results['pytorch_npu']:
            print("   - Check torch_npu installation")
        if not results['cpu']:
            print("   - Check NumPy installation")

if __name__ == "__main__":
    main()
