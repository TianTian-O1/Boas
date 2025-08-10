#!/usr/bin/env python3
"""
🚀 运行所有性能benchmark并生成完整报告
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_dependencies():
    """检查依赖项"""
    print("🔍 Checking dependencies...")
    
    dependencies = {
        'numpy': 'NumPy for CPU baseline',
        'matplotlib': 'Matplotlib for visualization', 
        'seaborn': 'Seaborn for better plots'
    }
    
    missing = []
    
    for dep, desc in dependencies.items():
        try:
            __import__(dep)
            print(f"✅ {dep} - {desc}")
        except ImportError:
            print(f"❌ {dep} - {desc}")
            missing.append(dep)
    
    # 检查torch_npu
    try:
        import torch
        import torch_npu
        print("✅ torch_npu - PyTorch NPU support")
    except ImportError:
        print("⚠️ torch_npu - PyTorch NPU support (optional)")
    
    if missing:
        print(f"\n📦 Installing missing dependencies: {', '.join(missing)}")
        subprocess.run([sys.executable, '-m', 'pip', 'install'] + missing)
    
    return True

def check_environment():
    """检查环境配置"""
    print("\n🔧 Checking environment...")
    
    # 检查CANN环境
    cann_path = "/usr/local/Ascend/ascend-toolkit/latest"
    if os.path.exists(cann_path):
        print(f"✅ CANN toolkit found at {cann_path}")
    else:
        print(f"❌ CANN toolkit not found at {cann_path}")
        return False
    
    # 检查Boas编译器
    boas_root = "/root/Boas/Boas-linux"
    pipeline_exe = os.path.join(boas_root, "build", "test-full-pipeline")
    
    if os.path.exists(pipeline_exe):
        print(f"✅ Boas compiler found at {pipeline_exe}")
    else:
        print(f"❌ Boas compiler not found. Please compile first.")
        return False
    
    # 检查NPU设备
    npu_devices = [f for f in os.listdir("/dev") if f.startswith("davinci")]
    if npu_devices:
        print(f"✅ NPU devices found: {', '.join(npu_devices)}")
    else:
        print("⚠️ No NPU devices found")
    
    return True

def run_benchmark_suite():
    """运行完整的benchmark套件"""
    print("\n🚀 Starting comprehensive benchmark suite...")
    print("=" * 80)
    
    benchmark_dir = Path(__file__).parent
    
    # 1. 运行主要benchmark
    print("📊 Phase 1: Running main performance benchmark...")
    main_benchmark = benchmark_dir / "boas_benchmark.py"
    
    if main_benchmark.exists():
        result = subprocess.run([sys.executable, str(main_benchmark)], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Main benchmark completed successfully")
            print(result.stdout[-500:])  # 显示最后500个字符
        else:
            print("❌ Main benchmark failed:")
            print(result.stderr[-500:])
            return False
    else:
        print(f"❌ Main benchmark script not found: {main_benchmark}")
        return False
    
    # 2. 运行Triton-Ascend对比 (可选)
    print("\n⚡ Phase 2: Running Triton-Ascend comparison...")
    triton_benchmark = benchmark_dir / "triton_ascend_benchmark.py"
    
    if triton_benchmark.exists():
        result = subprocess.run([sys.executable, str(triton_benchmark)], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Triton-Ascend benchmark completed")
        else:
            print("⚠️ Triton-Ascend benchmark failed (this is optional)")
            print(result.stderr[-300:])
    
    # 3. 生成可视化报告
    print("\n📊 Phase 3: Generating visualization and reports...")
    visualizer = benchmark_dir / "performance_visualizer.py"
    
    if visualizer.exists():
        result = subprocess.run([sys.executable, str(visualizer)], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Visualization and reports generated")
            print(result.stdout)
        else:
            print("❌ Visualization generation failed:")
            print(result.stderr[-300:])
    
    return True

def create_summary_report():
    """创建总结报告"""
    print("\n📝 Creating summary report...")
    
    summary = f"""
# 🚀 Boas Language NPU Benchmark Summary

**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}

## 🎯 Benchmark Overview

This comprehensive benchmark compares the performance of:

1. **🎯 Boas+CANN** - Our MLIR-based NPU compiler
2. **🔥 PyTorch+NPU** - Official torch_npu implementation  
3. **⚡ Triton-Ascend** - Triton-based NPU kernels
4. **💻 CPU Baseline** - NumPy CPU implementation

## 📊 Test Matrix

| Scale | Matrix Size | Memory | Compute |
|-------|-------------|--------|---------|
| Small | 64×64 | ~32KB | 262K FLOP |
| Medium | 256×256 | ~512KB | 33M FLOP |
| Large | 1024×1024 | ~8MB | 2.1B FLOP |
| XLarge | 2048×2048 | ~32MB | 17B FLOP |

## 🔧 Environment

- **Hardware**: 昇腾910B2 NPU
- **Software**: CANN 8.1.RC1, LLVM 20
- **Boas Version**: Latest with CANN integration
- **Comparison**: PyTorch 2.0 + torch_npu

## 📈 Key Metrics

- **Execution Time** (milliseconds)
- **Throughput** (GFLOPS)  
- **Memory Bandwidth** (GB/s)
- **NPU Utilization** (%)
- **Compilation Time** (seconds)

## 🏆 Expected Results

Based on our CANN integration:

- **Boas+CANN** should achieve competitive performance with PyTorch
- **Compilation time** advantage due to static optimization
- **Memory efficiency** from MLIR optimization passes
- **Scalability** benefits for large matrices

## 📂 Generated Files

- `benchmark_results.json` - Raw performance data
- `performance_comparison_*.png` - Performance plots
- `speedup_analysis_*.png` - Speedup analysis
- `performance_report_*.md` - Detailed markdown report

## 🔍 Analysis Focus

1. **Performance Parity**: How does Boas compare to PyTorch?
2. **Compilation Benefits**: Static vs dynamic optimization
3. **Scalability**: Performance across matrix sizes  
4. **NPU Utilization**: Hardware efficiency metrics
5. **Future Optimization**: Areas for improvement

---

🎯 **Goal**: Demonstrate that Boas language can achieve competitive NPU performance
through its MLIR-based CANN integration while providing better compile-time optimization.
"""
    
    with open("benchmark_summary.md", "w") as f:
        f.write(summary)
    
    print("✅ Summary report saved to benchmark_summary.md")

def main():
    """主函数"""
    print("🚀 Boas Language NPU Comprehensive Benchmark Suite")
    print("=" * 80)
    print("🎯 Comparing: Boas+CANN vs PyTorch+NPU vs Triton vs CPU")
    print("=" * 80)
    
    try:
        # 检查依赖
        if not check_dependencies():
            print("❌ Dependency check failed")
            sys.exit(1)
        
        # 检查环境
        if not check_environment():
            print("❌ Environment check failed")
            sys.exit(1)
        
        # 创建总结报告
        create_summary_report()
        
        # 运行benchmark套件
        success = run_benchmark_suite()
        
        if success:
            print("\n" + "=" * 80)
            print("🎉 BENCHMARK SUITE COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print("\n📂 Generated files:")
            
            # 列出生成的文件
            generated_files = [
                "benchmark_results.json",
                "benchmark_summary.md"
            ]
            
            # 查找时间戳文件
            import glob
            generated_files.extend(glob.glob("performance_comparison_*.png"))
            generated_files.extend(glob.glob("speedup_analysis_*.png"))
            generated_files.extend(glob.glob("performance_report_*.md"))
            
            for file in generated_files:
                if os.path.exists(file):
                    print(f"   ✅ {file}")
                else:
                    print(f"   ❌ {file} (not generated)")
            
            print("\n🔍 Check the generated reports for detailed performance analysis!")
            print("🚀 Boas language NPU performance evaluation complete!")
            
        else:
            print("\n❌ Benchmark suite failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️ Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
