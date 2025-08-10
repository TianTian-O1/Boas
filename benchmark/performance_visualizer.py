#!/usr/bin/env python3
"""
📊 性能可视化和报告生成器
"""

import json
import numpy as np
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class PerformanceVisualizer:
    """性能数据可视化"""
    
    def __init__(self, results_file: str = "benchmark_results.json"):
        self.results_file = results_file
        self.results = self.load_results()
        
        # 设置绘图样式
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def load_results(self) -> List[Dict[str, Any]]:
        """加载benchmark结果"""
        try:
            with open(self.results_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ Results file {self.results_file} not found")
            return []
    
    def organize_data(self) -> Dict[str, Dict[str, List]]:
        """整理数据按框架分组"""
        organized = {}
        
        for result in self.results:
            if not result['success']:
                continue
                
            framework = result['framework']
            if framework not in organized:
                organized[framework] = {
                    'matrix_sizes': [],
                    'execution_times': [],
                    'gflops': [],
                    'memory_bandwidth': [],
                    'npu_utilization': []
                }
            
            organized[framework]['matrix_sizes'].append(result['matrix_size'])
            organized[framework]['execution_times'].append(result['execution_time_ms'])
            organized[framework]['gflops'].append(result['gflops'])
            organized[framework]['memory_bandwidth'].append(result['memory_bandwidth_gb_s'])
            organized[framework]['npu_utilization'].append(result.get('npu_utilization', 0))
        
        return organized
    
    def plot_performance_comparison(self):
        """绘制性能对比图"""
        data = self.organize_data()
        
        if not data:
            print("❌ No data to visualize")
            return
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('🚀 Boas Language NPU Performance Benchmark', fontsize=16, fontweight='bold')
        
        # 1. 执行时间对比
        ax1 = axes[0, 0]
        for framework, values in data.items():
            ax1.plot(values['matrix_sizes'], values['execution_times'], 
                    marker='o', linewidth=2, label=framework)
        ax1.set_xlabel('Matrix Size')
        ax1.set_ylabel('Execution Time (ms)')
        ax1.set_title('⏱️ Execution Time Comparison')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. GFLOPS对比
        ax2 = axes[0, 1]
        for framework, values in data.items():
            ax2.plot(values['matrix_sizes'], values['gflops'], 
                    marker='s', linewidth=2, label=framework)
        ax2.set_xlabel('Matrix Size')
        ax2.set_ylabel('Performance (GFLOPS)')
        ax2.set_title('🚀 Performance (GFLOPS) Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 内存带宽对比
        ax3 = axes[1, 0]
        for framework, values in data.items():
            ax3.plot(values['matrix_sizes'], values['memory_bandwidth'], 
                    marker='^', linewidth=2, label=framework)
        ax3.set_xlabel('Matrix Size')
        ax3.set_ylabel('Memory Bandwidth (GB/s)')
        ax3.set_title('💾 Memory Bandwidth Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. NPU利用率
        ax4 = axes[1, 1]
        frameworks_with_npu = {}
        for framework, values in data.items():
            if any(util > 0 for util in values['npu_utilization']):
                frameworks_with_npu[framework] = values
        
        if frameworks_with_npu:
            for framework, values in frameworks_with_npu.items():
                ax4.bar(np.arange(len(values['matrix_sizes'])) + 
                       list(frameworks_with_npu.keys()).index(framework) * 0.2,
                       values['npu_utilization'], 
                       width=0.2, label=framework, alpha=0.8)
            ax4.set_xlabel('Matrix Size')
            ax4.set_ylabel('NPU Utilization (%)')
            ax4.set_title('🎯 NPU Utilization')
            ax4.set_xticks(np.arange(len(values['matrix_sizes'])))
            ax4.set_xticklabels([str(size) for size in values['matrix_sizes']])
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No NPU utilization data', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('🎯 NPU Utilization (No Data)')
        
        plt.tight_layout()
        
        # 保存图片
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"performance_comparison_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 Performance comparison plot saved to {filename}")
        
        return fig
    
    def plot_speedup_analysis(self):
        """绘制加速比分析"""
        data = self.organize_data()
        
        if len(data) < 2:
            print("❌ Need at least 2 frameworks for speedup analysis")
            return
        
        # 使用CPU作为基准
        cpu_framework = None
        for framework in data.keys():
            if 'CPU' in framework.upper():
                cpu_framework = framework
                break
        
        if not cpu_framework:
            print("❌ No CPU baseline found for speedup analysis")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('🚀 NPU Speedup Analysis vs CPU Baseline', fontsize=14, fontweight='bold')
        
        cpu_data = data[cpu_framework]
        matrix_sizes = cpu_data['matrix_sizes']
        
        # 1. 性能加速比
        for framework, values in data.items():
            if framework == cpu_framework:
                continue
            
            speedup = np.array(values['gflops']) / np.array(cpu_data['gflops'])
            ax1.plot(matrix_sizes, speedup, marker='o', linewidth=2, label=framework)
        
        ax1.set_xlabel('Matrix Size')
        ax1.set_ylabel('Speedup (vs CPU)')
        ax1.set_title('🏃‍♂️ Performance Speedup')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='CPU Baseline')
        
        # 2. 时间减少比例
        for framework, values in data.items():
            if framework == cpu_framework:
                continue
            
            time_reduction = (1 - np.array(values['execution_times']) / np.array(cpu_data['execution_times'])) * 100
            ax2.plot(matrix_sizes, time_reduction, marker='s', linewidth=2, label=framework)
        
        ax2.set_xlabel('Matrix Size')
        ax2.set_ylabel('Time Reduction (%)')
        ax2.set_title('⏰ Time Reduction vs CPU')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='CPU Baseline')
        
        plt.tight_layout()
        
        # 保存图片
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"speedup_analysis_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 Speedup analysis plot saved to {filename}")
        
        return fig
    
    def generate_performance_table(self) -> str:
        """生成性能对比表格"""
        data = self.organize_data()
        
        if not data:
            return "❌ No data available for table generation"
        
        # 创建markdown表格
        table = []
        table.append("# 🚀 Boas Language NPU Performance Benchmark Results")
        table.append("")
        table.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        table.append("")
        
        # 按矩阵大小组织数据
        all_sizes = set()
        for framework_data in data.values():
            all_sizes.update(framework_data['matrix_sizes'])
        all_sizes = sorted(all_sizes)
        
        # 性能表格
        table.append("## 📊 Performance Comparison (GFLOPS)")
        table.append("")
        
        # 表头
        header = "| Matrix Size |"
        separator = "|-------------|"
        for framework in data.keys():
            header += f" {framework} |"
            separator += "----------|"
        table.append(header)
        table.append(separator)
        
        # 数据行
        for size in all_sizes:
            row = f"| {size}×{size} |"
            for framework, values in data.items():
                if size in values['matrix_sizes']:
                    idx = values['matrix_sizes'].index(size)
                    gflops = values['gflops'][idx]
                    row += f" {gflops:.1f} |"
                else:
                    row += " N/A |"
            table.append(row)
        
        table.append("")
        
        # 执行时间表格
        table.append("## ⏱️ Execution Time Comparison (ms)")
        table.append("")
        table.append(header)
        table.append(separator)
        
        for size in all_sizes:
            row = f"| {size}×{size} |"
            for framework, values in data.items():
                if size in values['matrix_sizes']:
                    idx = values['matrix_sizes'].index(size)
                    time_ms = values['execution_times'][idx]
                    row += f" {time_ms:.2f} |"
                else:
                    row += " N/A |"
            table.append(row)
        
        table.append("")
        
        # 性能排名
        table.append("## 🏆 Performance Ranking")
        table.append("")
        
        # 计算每个框架的平均性能
        avg_performance = {}
        for framework, values in data.items():
            avg_performance[framework] = np.mean(values['gflops'])
        
        # 排序
        ranked = sorted(avg_performance.items(), key=lambda x: x[1], reverse=True)
        
        for i, (framework, avg_gflops) in enumerate(ranked):
            table.append(f"{i+1}. **{framework}**: {avg_gflops:.1f} GFLOPS (average)")
        
        table.append("")
        table.append("## 📈 Key Insights")
        table.append("")
        
        if len(ranked) >= 2:
            best = ranked[0]
            second = ranked[1]
            improvement = (best[1] / second[1] - 1) * 100
            table.append(f"- **{best[0]}** achieves the best performance with {best[1]:.1f} GFLOPS average")
            table.append(f"- **{best[0]}** is {improvement:.1f}% faster than **{second[0]}** on average")
        
        # 检查Boas性能
        boas_frameworks = [f for f in data.keys() if 'Boas' in f]
        if boas_frameworks:
            boas_framework = boas_frameworks[0]
            boas_rank = [i for i, (f, _) in enumerate(ranked) if f == boas_framework][0] + 1
            boas_perf = avg_performance[boas_framework]
            table.append(f"- **Boas+CANN** ranks #{boas_rank} with {boas_perf:.1f} GFLOPS average")
            
            # 与PyTorch对比
            pytorch_frameworks = [f for f in data.keys() if 'PyTorch' in f]
            if pytorch_frameworks:
                pytorch_framework = pytorch_frameworks[0]
                pytorch_perf = avg_performance[pytorch_framework]
                ratio = boas_perf / pytorch_perf
                if ratio > 1:
                    table.append(f"- **Boas+CANN** is {ratio:.2f}x faster than **{pytorch_framework}**")
                else:
                    table.append(f"- **Boas+CANN** is {1/ratio:.2f}x slower than **{pytorch_framework}**")
        
        return "\n".join(table)
    
    def save_report(self, filename: str = None):
        """保存完整的性能报告"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_report_{timestamp}.md"
        
        report = self.generate_performance_table()
        
        with open(filename, 'w') as f:
            f.write(report)
        
        print(f"📝 Performance report saved to {filename}")
        return filename

def main():
    """主函数"""
    print("📊 Performance Visualization and Reporting")
    print("=" * 50)
    
    visualizer = PerformanceVisualizer()
    
    if not visualizer.results:
        print("❌ No benchmark results found. Run boas_benchmark.py first.")
        return
    
    try:
        # 生成性能对比图
        print("🎨 Generating performance comparison plots...")
        visualizer.plot_performance_comparison()
        
        # 生成加速比分析
        print("🚀 Generating speedup analysis...")
        visualizer.plot_speedup_analysis()
        
        # 生成性能报告
        print("📝 Generating performance report...")
        visualizer.save_report()
        
        print("✅ All visualizations and reports generated successfully!")
        
    except Exception as e:
        print(f"❌ Error generating visualizations: {e}")

if __name__ == "__main__":
    main()
