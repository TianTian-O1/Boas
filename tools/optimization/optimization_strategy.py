#!/usr/bin/env python3
"""
🚀 Boas语言NPU性能优化策略分析器
分析当前性能瓶颈，制定优化计划，并实施优化措施
"""

import subprocess
import time
import numpy as np
import os
import json
from datetime import datetime

class BoasOptimizer:
    def __init__(self):
        self.target_performance = {
            'minimum': 3220,  # GFLOPS (80% of PyTorch)
            'competitive': 4026,  # GFLOPS (100% of PyTorch)
            'excellent': 4831   # GFLOPS (120% of PyTorch)
        }
        
        self.optimization_areas = [
            "MLIR优化passes",
            "内存布局优化", 
            "NPU并行策略",
            "数据类型优化",
            "算法选择优化"
        ]
        
    def analyze_current_status(self):
        """分析当前Boas性能状态"""
        print("🔍 当前性能状态分析")
        print("=" * 50)
        
        # 检查可执行文件
        if os.path.exists("boas_npu_test"):
            print("✅ 可执行文件存在: boas_npu_test")
            file_size = os.path.getsize("boas_npu_test")
            print(f"📄 文件大小: {file_size} bytes")
        else:
            print("❌ 可执行文件不存在")
            return False
            
        # 检查MLIR文件
        if os.path.exists("temp.final.ll"):
            print("✅ LLVM IR文件存在")
            with open("temp.final.ll", 'r') as f:
                lines = len(f.readlines())
            print(f"📄 LLVM IR行数: {lines}")
        else:
            print("❌ LLVM IR文件不存在")
            
        # 分析NPU利用率
        print("\n🎯 NPU状态分析:")
        try:
            result = subprocess.run(['npu-smi', 'info'], 
                                 capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'NPU' in line and ('OK' in line or '%' in line):
                        print(f"   {line.strip()}")
        except:
            print("   ⚠️ 无法获取NPU状态")
            
        return True
        
    def benchmark_current_performance(self):
        """基准测试当前性能"""
        print("\n🚀 当前性能基准测试")
        print("=" * 50)
        
        print("🔧 准备测试环境...")
        # 设置环境变量
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = "/usr/lib/aarch64-linux-gnu:/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64:" + env.get("LD_LIBRARY_PATH", "")
        
        # 运行基本性能测试
        print("⏱️ 执行基础矩阵乘法测试...")
        
        times = []
        for i in range(5):
            start_time = time.time()
            try:
                result = subprocess.run(['./boas_npu_test'], 
                                      capture_output=True, text=True, 
                                      timeout=10, env=env)
                end_time = time.time()
                
                if result.returncode == 0:
                    execution_time = (end_time - start_time) * 1000  # ms
                    times.append(execution_time)
                    print(f"   测试 {i+1}: {execution_time:.3f} ms")
                else:
                    print(f"   测试 {i+1}: 执行失败")
            except subprocess.TimeoutExpired:
                print(f"   测试 {i+1}: 超时")
            except Exception as e:
                print(f"   测试 {i+1}: 错误 - {e}")
                
        if times:
            avg_time = np.mean(times)
            std_time = np.std(times)
            
            # 估算性能 (2x2矩阵，2*2^3 = 16 FLOP)
            flops = 2 * 2 * 2 * 2  # 简单估算
            gflops = flops / (avg_time / 1000) / 1e9
            
            print(f"\n📊 性能结果:")
            print(f"   平均执行时间: {avg_time:.3f} ± {std_time:.3f} ms")
            print(f"   估算性能: {gflops:.3f} GFLOPS")
            
            return {
                'avg_time_ms': avg_time,
                'std_time_ms': std_time, 
                'estimated_gflops': gflops
            }
        else:
            print("❌ 所有测试都失败了")
            return None
            
    def identify_optimization_opportunities(self, current_perf=None):
        """识别优化机会"""
        print("\n🎯 优化机会分析")
        print("=" * 50)
        
        opportunities = []
        
        # 1. MLIR优化分析
        print("🔍 1. MLIR编译优化分析:")
        if os.path.exists("temp.llvm.mlir"):
            with open("temp.llvm.mlir", 'r') as f:
                mlir_content = f.read()
                
            # 检查是否有优化机会
            if 'scf.for' in mlir_content:
                print("   ⚡ 发现SCF循环 - 可以向量化优化")
                opportunities.append("scf_vectorization")
                
            if 'linalg.matmul' in mlir_content:
                print("   🎯 发现linalg.matmul - 可以应用tiling优化") 
                opportunities.append("linalg_tiling")
                
            if 'memref.alloc' in mlir_content:
                print("   💾 发现内存分配 - 可以内存池优化")
                opportunities.append("memory_pool")
        
        # 2. 算法优化分析  
        print("\n🔍 2. 算法优化分析:")
        print("   🧮 当前使用朴素三重循环")
        print("   ⚡ 可以应用分块算法(Blocking)")
        print("   🔀 可以应用循环重排序") 
        opportunities.extend(["matrix_blocking", "loop_reordering"])
        
        # 3. NPU特化优化
        print("\n🔍 3. NPU特化优化:")
        print("   🎯 可以应用对角分块(Diagonal Tiling)")
        print("   🔧 可以优化AI Core并行度")
        print("   📊 可以应用混合精度(FP16/BF16)")
        opportunities.extend(["diagonal_tiling", "multi_core", "mixed_precision"])
        
        self.opportunities = opportunities
        return opportunities
        
    def create_optimization_plan(self, opportunities):
        """创建优化实施计划"""
        print("\n📋 优化实施计划")
        print("=" * 50)
        
        plan = {
            "phase1_immediate": [
                ("MLIR优化passes", "添加向量化和循环优化", "高"),
                ("内存布局优化", "改进数据访问模式", "高"),
                ("编译器标志", "启用最高优化级别", "中")
            ],
            "phase2_algorithmic": [
                ("矩阵分块", "实现分块矩阵乘法", "高"),
                ("循环优化", "循环重排序和展开", "中"),
                ("并行化", "多AI Core并行", "高")
            ],
            "phase3_advanced": [
                ("混合精度", "FP16/BF16优化", "中"),
                ("内存预取", "优化内存访问", "中"),
                ("kernel融合", "算子融合优化", "低")
            ]
        }
        
        for phase, tasks in plan.items():
            print(f"\n🎯 {phase.upper()}:")
            for i, (task, desc, priority) in enumerate(tasks, 1):
                priority_icon = "🔥" if priority == "高" else "⚡" if priority == "中" else "💡"
                print(f"   {i}. {priority_icon} {task}: {desc}")
                
        return plan
        
    def implement_immediate_optimizations(self):
        """实施立即可用的优化"""
        print("\n🚀 实施立即优化")
        print("=" * 50)
        
        optimizations_applied = []
        
        # 1. 改进MLIR优化pipeline
        print("🔧 1. 改进MLIR优化pipeline...")
        
        # 创建优化版本的转换脚本
        optimized_pipeline = '''#!/bin/bash
# 🚀 优化版MLIR转换pipeline

echo "🔧 应用高级MLIR优化..."

/usr/local/llvm-20/bin/mlir-opt temp.llvm.mlir \\
    --convert-linalg-to-loops \\
    --loop-invariant-code-motion \\
    --affine-loop-unroll="unroll-factor=8" \\
    --affine-loop-fusion \\
    --canonicalize \\
    --cse \\
    --convert-scf-to-cf \\
    --convert-cf-to-llvm \\
    --reconcile-unrealized-casts \\
    -o temp.optimized.mlir

if [ $? -eq 0 ]; then
    echo "✅ 优化pipeline成功"
    
    echo "🔧 生成优化的LLVM IR..."
    /usr/local/llvm-20/bin/mlir-translate \\
        --mlir-to-llvmir temp.optimized.mlir \\
        -o temp.optimized.ll \\
        --allow-unregistered-dialect
        
    if [ $? -eq 0 ]; then
        echo "✅ 优化LLVM IR生成成功"
        
        echo "🔧 生成优化汇编(-O3 + 向量化)..."
        /usr/local/llvm-20/bin/llc temp.optimized.ll \\
            -o temp.optimized.s \\
            -O3 \\
            -enable-unsafe-fp-math \\
            -ffast-math
            
        if [ $? -eq 0 ]; then
            echo "✅ 优化汇编生成成功"
            
            echo "🔧 链接优化版本..."
            gcc temp.optimized.s -o boas_npu_optimized \\
                -O3 -ffast-math -march=native \\
                -L/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64 \\
                -lm
                
            if [ $? -eq 0 ]; then
                echo "🎉 优化版本编译成功: boas_npu_optimized"
                ls -la boas_npu_optimized
            else
                echo "❌ 链接失败"
            fi
        else
            echo "❌ 汇编生成失败"
        fi
    else
        echo "❌ LLVM IR生成失败"
    fi
else
    echo "❌ 优化pipeline失败"
fi
'''
        
        with open("optimize_compile.sh", 'w') as f:
            f.write(optimized_pipeline)
        os.chmod("optimize_compile.sh", 0o755)
        
        print("📁 创建了优化编译脚本: optimize_compile.sh")
        optimizations_applied.append("optimized_compilation_pipeline")
        
        # 2. 执行优化编译
        print("\n🚀 执行优化编译...")
        try:
            result = subprocess.run(['./optimize_compile.sh'], 
                                  capture_output=True, text=True, timeout=60)
            print(result.stdout)
            if result.stderr:
                print("stderr:", result.stderr)
                
            if os.path.exists("boas_npu_optimized"):
                print("✅ 优化版本编译成功")
                optimizations_applied.append("optimized_executable")
            else:
                print("❌ 优化版本编译失败")
        except Exception as e:
            print(f"❌ 优化编译错误: {e}")
        
        return optimizations_applied
        
    def benchmark_optimized_performance(self):
        """测试优化后性能"""
        print("\n📊 优化后性能测试")
        print("=" * 50)
        
        results = {}
        
        # 测试原版本
        if os.path.exists("boas_npu_test"):
            print("🔍 测试原版本性能...")
            orig_perf = self.run_performance_test("boas_npu_test", "原版本")
            results['original'] = orig_perf
        
        # 测试优化版本
        if os.path.exists("boas_npu_optimized"):
            print("\n🚀 测试优化版本性能...")
            opt_perf = self.run_performance_test("boas_npu_optimized", "优化版本")
            results['optimized'] = opt_perf
            
            # 计算性能提升
            if 'original' in results and results['original'] and opt_perf:
                speedup = results['original']['avg_time_ms'] / opt_perf['avg_time_ms']
                print(f"\n🎯 性能提升: {speedup:.2f}x")
                
                # 估算GFLOPS提升
                if 'estimated_gflops' in results['original'] and 'estimated_gflops' in opt_perf:
                    gflops_improvement = opt_perf['estimated_gflops'] - results['original']['estimated_gflops']
                    print(f"🚀 GFLOPS提升: +{gflops_improvement:.3f}")
        
        return results
        
    def run_performance_test(self, executable, name):
        """运行性能测试"""
        print(f"   ⏱️ 测试 {name}...")
        
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = "/usr/lib/aarch64-linux-gnu:/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64:" + env.get("LD_LIBRARY_PATH", "")
        
        times = []
        for i in range(10):  # 更多测试次数获得准确结果
            start_time = time.time()
            try:
                result = subprocess.run([f'./{executable}'], 
                                      capture_output=True, text=True, 
                                      timeout=5, env=env)
                end_time = time.time()
                
                if result.returncode == 0:
                    execution_time = (end_time - start_time) * 1000  # ms
                    times.append(execution_time)
                    if i < 3:  # 只显示前几次
                        print(f"     运行 {i+1}: {execution_time:.3f} ms")
                    elif i == 3:
                        print(f"     ... (继续测试)")
                        
            except subprocess.TimeoutExpired:
                print(f"     运行 {i+1}: 超时")
            except Exception as e:
                print(f"     运行 {i+1}: 错误")
                
        if times:
            avg_time = np.mean(times)
            std_time = np.std(times)
            min_time = np.min(times)
            
            # 2x2矩阵乘法的FLOPS估算
            flops = 2 * 2 * 2 * 2  # 非常简单的测试
            gflops = flops / (avg_time / 1000) / 1e9
            
            print(f"   📊 {name}结果:")
            print(f"      平均时间: {avg_time:.3f} ± {std_time:.3f} ms")
            print(f"      最佳时间: {min_time:.3f} ms") 
            print(f"      估算性能: {gflops:.6f} GFLOPS")
            
            return {
                'avg_time_ms': avg_time,
                'std_time_ms': std_time,
                'min_time_ms': min_time,
                'estimated_gflops': gflops,
                'name': name
            }
        else:
            print(f"   ❌ {name}测试失败")
            return None
            
    def generate_optimization_report(self, results):
        """生成优化报告"""
        print("\n📋 优化报告生成")
        print("=" * 50)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'optimization_results': results,
            'target_performance': self.target_performance,
            'recommendations': []
        }
        
        # 分析结果
        if 'optimized' in results and results['optimized']:
            current_gflops = results['optimized']['estimated_gflops']
            
            if current_gflops >= self.target_performance['excellent']:
                status = "🏆 卓越性能"
                report['recommendations'].append("性能已超越目标，考虑更大规模测试")
            elif current_gflops >= self.target_performance['competitive']:
                status = "🥈 竞争性能"
                report['recommendations'].append("已达到竞争目标，可进一步优化算法")
            elif current_gflops >= self.target_performance['minimum']:
                status = "🥉 最低目标"
                report['recommendations'].append("达到最低目标，需要更多优化")
            else:
                status = "⚠️ 需要改进"
                report['recommendations'].extend([
                    "考虑更大的测试矩阵",
                    "实施算法级优化",
                    "检查NPU utilization"
                ])
                
            print(f"🎯 当前状态: {status}")
            print(f"📊 当前性能: {current_gflops:.6f} GFLOPS")
            
        # 保存报告
        with open('optimization_report.json', 'w') as f:
            json.dump(report, f, indent=2)
            
        print("📁 优化报告已保存: optimization_report.json")
        return report

def main():
    """主优化流程"""
    print("🚀 Boas语言NPU性能优化器启动")
    print("=" * 60)
    
    optimizer = BoasOptimizer()
    
    # 1. 分析当前状态
    if not optimizer.analyze_current_status():
        print("❌ 当前状态检查失败，请先完成编译")
        return
        
    # 2. 基准测试
    current_perf = optimizer.benchmark_current_performance()
    
    # 3. 识别优化机会
    opportunities = optimizer.identify_optimization_opportunities(current_perf)
    
    # 4. 创建优化计划
    plan = optimizer.create_optimization_plan(opportunities)
    
    # 5. 实施立即优化
    applied_opts = optimizer.implement_immediate_optimizations()
    
    # 6. 测试优化后性能
    perf_results = optimizer.benchmark_optimized_performance()
    
    # 7. 生成报告
    report = optimizer.generate_optimization_report(perf_results)
    
    print("\n🎉 优化流程完成!")
    print("📋 查看详细报告: optimization_report.json")
    
    # 下一步建议
    print("\n🔮 下一步建议:")
    for rec in report.get('recommendations', []):
        print(f"   • {rec}")

if __name__ == "__main__":
    main()
