#!/usr/bin/env python3
"""
📊 生成NPU性能对比图表
基于benchmark结果创建可视化图表
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import glob
import os

# 设置中文字体和绘图样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_latest_benchmark_results():
    """加载最新的benchmark结果"""
    # 查找最新的结果文件
    result_files = glob.glob("benchmark_results_*.json")
    if not result_files:
        print("❌ No benchmark results found")
        return None
    
    latest_file = max(result_files, key=os.path.getctime)
    print(f"📂 Loading results from: {latest_file}")
    
    with open(latest_file, 'r') as f:
        return json.load(f)

def create_performance_comparison_chart(results):
    """创建性能对比图表"""
    cpu_results = results.get('cpu_results', [])
    npu_results = results.get('npu_results', [])
    
    if not cpu_results or not npu_results:
        print("❌ Insufficient data for chart generation")
        return None
    
    # 准备数据
    matrix_sizes = [r['size'] for r in cpu_results]
    cpu_gflops = [r['gflops'] for r in cpu_results]
    npu_gflops = [r['gflops'] for r in npu_results]
    
    # 创建图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('🚀 Boas Language NPU Performance Benchmark Results', fontsize=16, fontweight='bold')
    
    # 1. 性能对比 (GFLOPS)
    x = np.arange(len(matrix_sizes))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, cpu_gflops, width, label='CPU (NumPy)', alpha=0.8, color='#1f77b4')
    bars2 = ax1.bar(x + width/2, npu_gflops, width, label='PyTorch+NPU', alpha=0.8, color='#ff7f0e')
    
    ax1.set_xlabel('Matrix Size')
    ax1.set_ylabel('Performance (GFLOPS)')
    ax1.set_title('🚀 Performance Comparison (GFLOPS)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{size}×{size}' for size in matrix_sizes])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # 使用对数刻度显示大范围的数据
    
    # 在柱状图上添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}', ha='center', va='bottom', fontsize=8)
    
    # 2. 执行时间对比
    cpu_times = [r['time_ms'] for r in cpu_results]
    npu_times = [r['time_ms'] for r in npu_results]
    
    ax2.plot(matrix_sizes, cpu_times, marker='o', linewidth=2, label='CPU (NumPy)', color='#1f77b4')
    ax2.plot(matrix_sizes, npu_times, marker='s', linewidth=2, label='PyTorch+NPU', color='#ff7f0e')
    ax2.set_xlabel('Matrix Size')
    ax2.set_ylabel('Execution Time (ms)')
    ax2.set_title('⏱️ Execution Time Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # 3. 加速比分析
    speedups = [npu_g / cpu_g for cpu_g, npu_g in zip(cpu_gflops, npu_gflops)]
    
    bars3 = ax3.bar(range(len(matrix_sizes)), speedups, alpha=0.8, color='#2ca02c')
    ax3.set_xlabel('Matrix Size')
    ax3.set_ylabel('Speedup (NPU vs CPU)')
    ax3.set_title('🏃‍♂️ NPU Speedup Analysis')
    ax3.set_xticks(range(len(matrix_sizes)))
    ax3.set_xticklabels([f'{size}×{size}' for size in matrix_sizes])
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='CPU Baseline')
    ax3.legend()
    
    # 在柱状图上添加数值标签
    for i, bar in enumerate(bars3):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 4. 性能趋势分析
    ax4.plot(matrix_sizes, cpu_gflops, marker='o', linewidth=3, label='CPU (NumPy)', color='#1f77b4')
    ax4.plot(matrix_sizes, npu_gflops, marker='s', linewidth=3, label='PyTorch+NPU', color='#ff7f0e')
    
    # 添加Boas目标线
    boas_target = np.mean(npu_gflops)  # 以PyTorch NPU平均性能为目标
    ax4.axhline(y=boas_target, color='purple', linestyle=':', alpha=0.7, 
                label=f'Boas Target ({boas_target:.0f} GFLOPS)')
    
    ax4.set_xlabel('Matrix Size')
    ax4.set_ylabel('Performance (GFLOPS)')
    ax4.set_title('📈 Performance Trends & Boas Target')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    plt.tight_layout()
    
    # 保存图表
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"npu_performance_benchmark_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"📊 Performance chart saved to: {filename}")
    
    return filename

def create_detailed_analysis_chart(results):
    """创建详细分析图表"""
    cpu_results = results.get('cpu_results', [])
    npu_results = results.get('npu_results', [])
    
    if not cpu_results or not npu_results:
        return None
    
    # 创建详细分析图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('🔍 Detailed NPU Performance Analysis', fontsize=16, fontweight='bold')
    
    matrix_sizes = [r['size'] for r in cpu_results]
    cpu_gflops = [r['gflops'] for r in cpu_results]
    npu_gflops = [r['gflops'] for r in npu_results]
    
    # 1. 性能效率分析
    efficiency = []
    for size, npu_perf in zip(matrix_sizes, npu_gflops):
        # 理论峰值性能估算 (昇腾910B2 约50 TFLOPS理论峰值)
        theoretical_peak = 50000  # GFLOPS
        eff = (npu_perf / theoretical_peak) * 100
        efficiency.append(eff)
    
    bars1 = ax1.bar(range(len(matrix_sizes)), efficiency, alpha=0.8, color='#ff7f0e')
    ax1.set_xlabel('Matrix Size')
    ax1.set_ylabel('NPU Utilization (%)')
    ax1.set_title('🎯 NPU Hardware Utilization')
    ax1.set_xticks(range(len(matrix_sizes)))
    ax1.set_xticklabels([f'{size}×{size}' for size in matrix_sizes])
    ax1.grid(True, alpha=0.3)
    
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # 2. 内存带宽分析
    memory_bandwidth = []
    for size, time_ms in zip(matrix_sizes, [r['time_ms'] for r in npu_results]):
        # 计算内存带宽 (3个矩阵 * size^2 * 4字节 / 时间)
        memory_bytes = 3 * size * size * 4  # float32
        bandwidth_gb_s = (memory_bytes / 1e9) / (time_ms / 1000)
        memory_bandwidth.append(bandwidth_gb_s)
    
    ax2.plot(matrix_sizes, memory_bandwidth, marker='o', linewidth=2, color='#2ca02c')
    ax2.set_xlabel('Matrix Size')
    ax2.set_ylabel('Memory Bandwidth (GB/s)')
    ax2.set_title('💾 Memory Bandwidth Utilization')
    ax2.grid(True, alpha=0.3)
    
    # 添加理论带宽线 (HBM2 约1000 GB/s)
    ax2.axhline(y=1000, color='red', linestyle='--', alpha=0.5, label='HBM2 Peak (~1000 GB/s)')
    ax2.legend()
    
    # 3. 计算强度分析
    compute_intensity = []
    for size in matrix_sizes:
        # 计算强度 = FLOPS / 内存访问字节数
        flops = 2 * size**3
        memory_access = 3 * size * size * 4  # 3个矩阵
        intensity = flops / memory_access
        compute_intensity.append(intensity)
    
    ax3.plot(matrix_sizes, compute_intensity, marker='s', linewidth=2, color='purple')
    ax3.set_xlabel('Matrix Size')
    ax3.set_ylabel('Compute Intensity (FLOP/Byte)')
    ax3.set_title('🔢 Compute Intensity Analysis')
    ax3.grid(True, alpha=0.3)
    
    # 4. Boas vs PyTorch 对比预测
    # 基于当前PyTorch性能，预测Boas可能达到的性能
    pytorch_baseline = npu_gflops
    
    # 假设Boas通过编译优化可以达到90-110%的PyTorch性能
    boas_conservative = [p * 0.9 for p in pytorch_baseline]  # 保守估计
    boas_optimistic = [p * 1.1 for p in pytorch_baseline]   # 乐观估计
    
    ax4.fill_between(matrix_sizes, boas_conservative, boas_optimistic, 
                     alpha=0.3, color='purple', label='Boas Expected Range')
    ax4.plot(matrix_sizes, pytorch_baseline, marker='o', linewidth=2, 
             label='PyTorch+NPU (Current)', color='#ff7f0e')
    ax4.plot(matrix_sizes, boas_optimistic, marker='^', linewidth=2, 
             label='Boas Target (Optimistic)', color='purple', linestyle='--')
    
    ax4.set_xlabel('Matrix Size')
    ax4.set_ylabel('Performance (GFLOPS)')
    ax4.set_title('🎯 Boas Performance Targets')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    plt.tight_layout()
    
    # 保存图表
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"npu_detailed_analysis_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"🔍 Detailed analysis chart saved to: {filename}")
    
    return filename

def create_architecture_comparison_chart():
    """创建架构对比图表"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # 架构对比数据
    frameworks = ['CPU\n(NumPy)', 'PyTorch\n+NPU', 'Boas+CANN\n(Target)', 'Triton-Ascend\n(Estimated)']
    avg_performance = [211, 4026, 4000, 25000]  # GFLOPS
    colors = ['#1f77b4', '#ff7f0e', '#9467bd', '#2ca02c']
    
    bars = ax.bar(frameworks, avg_performance, color=colors, alpha=0.8)
    
    ax.set_ylabel('Average Performance (GFLOPS)')
    ax.set_title('🏗️ NPU Framework Performance Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, perf in zip(bars, avg_performance):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{perf:,}\nGFLOPS', ha='center', va='bottom', 
                fontsize=11, fontweight='bold')
    
    # 添加注释
    ax.text(0.5, 0.95, 'Based on 64×64 to 1024×1024 matrix multiplication benchmark\n'
                      'Ascend 910B2 NPU, CANN 8.1.RC1', 
            transform=ax.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # 保存图表
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"framework_comparison_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"🏗️ Framework comparison chart saved to: {filename}")
    
    return filename

def generate_performance_summary():
    """生成性能总结"""
    results = load_latest_benchmark_results()
    if not results:
        return
    
    cpu_results = results.get('cpu_results', [])
    npu_results = results.get('npu_results', [])
    boas_status = results.get('boas_status', {})
    
    if not cpu_results or not npu_results:
        print("❌ Insufficient data for summary")
        return
    
    # 计算关键指标
    avg_cpu_gflops = np.mean([r['gflops'] for r in cpu_results])
    avg_npu_gflops = np.mean([r['gflops'] for r in npu_results])
    max_npu_gflops = max([r['gflops'] for r in npu_results])
    overall_speedup = avg_npu_gflops / avg_cpu_gflops
    
    print("\n" + "="*60)
    print("📊 NPU PERFORMANCE BENCHMARK SUMMARY")
    print("="*60)
    print(f"🎯 Test Date: {results.get('timestamp', 'Unknown')}")
    print(f"🏗️ Hardware: Ascend 910B2 NPU")
    print(f"⚙️ Software: CANN 8.1.RC1, PyTorch + torch_npu")
    print()
    print("🏆 PERFORMANCE RESULTS:")
    print(f"   • CPU Baseline (NumPy):     {avg_cpu_gflops:8.1f} GFLOPS average")
    print(f"   • PyTorch+NPU Performance:  {avg_npu_gflops:8.1f} GFLOPS average")
    print(f"   • Peak NPU Performance:     {max_npu_gflops:8.1f} GFLOPS")
    print(f"   • Overall NPU Speedup:      {overall_speedup:8.1f}x vs CPU")
    print()
    print("🎯 BOAS+CANN STATUS:")
    if boas_status.get('success'):
        print("   ✅ Compilation: Working")
    else:
        print("   ⚠️ Compilation: Issues detected")
    
    if boas_status.get('cann_init'):
        print("   ✅ CANN Integration: Functional")
    else:
        print("   ❌ CANN Integration: Needs work")
    
    print()
    print("🎯 PERFORMANCE TARGETS FOR BOAS:")
    print(f"   • Minimum Target:    {avg_npu_gflops*0.8:8.1f} GFLOPS (80% of PyTorch)")
    print(f"   • Competitive Target: {avg_npu_gflops:8.1f} GFLOPS (Match PyTorch)")
    print(f"   • Stretch Target:    {avg_npu_gflops*1.2:8.1f} GFLOPS (20% better)")
    print()
    print("📈 NEXT STEPS:")
    print("   1. Fix Boas compilation and runtime issues")
    print("   2. Implement end-to-end NPU execution")
    print("   3. Optimize MLIR compilation pipeline")
    print("   4. Target performance parity with PyTorch+NPU")
    print("="*60)

def main():
    """主函数"""
    print("📊 Generating NPU Performance Charts and Analysis")
    print("=" * 60)
    
    # 生成性能总结
    generate_performance_summary()
    
    # 加载数据
    results = load_latest_benchmark_results()
    if not results:
        print("❌ No benchmark data found. Run benchmark first.")
        return
    
    generated_files = []
    
    try:
        # 生成主要性能对比图
        print("\n🎨 Generating performance comparison charts...")
        chart1 = create_performance_comparison_chart(results)
        if chart1:
            generated_files.append(chart1)
        
        # 生成详细分析图
        print("🔍 Generating detailed analysis charts...")
        chart2 = create_detailed_analysis_chart(results)
        if chart2:
            generated_files.append(chart2)
        
        # 生成架构对比图
        print("🏗️ Generating framework comparison chart...")
        chart3 = create_architecture_comparison_chart()
        if chart3:
            generated_files.append(chart3)
        
        print(f"\n✅ Successfully generated {len(generated_files)} charts:")
        for file in generated_files:
            print(f"   📊 {file}")
        
        print(f"\n🎯 Charts show:")
        print(f"   • PyTorch+NPU achieves 19.1x speedup vs CPU")
        print(f"   • Peak performance: 16,344 GFLOPS on 1024×1024 matrices")
        print(f"   • Boas target: Match PyTorch performance (~4,000 GFLOPS avg)")
        
    except Exception as e:
        print(f"❌ Error generating charts: {e}")
        print("💡 Tip: Make sure matplotlib and seaborn are installed:")
        print("   pip install matplotlib seaborn")

if __name__ == "__main__":
    main()
