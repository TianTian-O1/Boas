#!/usr/bin/env python3
"""
📊 Benchmark性能对比分析器
深度分析Boas vs PyTorch vs CPU的性能差距和优化潜力
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

class BenchmarkAnalyzer:
    def __init__(self):
        self.latest_results = None
        self.load_latest_results()
        
    def load_latest_results(self):
        """加载最新的benchmark结果"""
        results_dir = "results/optimization"
        if not os.path.exists(results_dir):
            print("❌ 未找到benchmark结果目录")
            return
            
        # 查找最新的benchmark文件
        benchmark_files = [f for f in os.listdir(results_dir) if f.startswith('comprehensive_benchmark_')]
        if not benchmark_files:
            print("❌ 未找到benchmark结果文件")
            return
            
        latest_file = max(benchmark_files)
        file_path = os.path.join(results_dir, latest_file)
        
        try:
            with open(file_path, 'r') as f:
                self.latest_results = json.load(f)
            print(f"✅ 加载了最新benchmark结果: {latest_file}")
        except Exception as e:
            print(f"❌ 加载benchmark结果失败: {e}")
            
    def analyze_performance_scaling(self):
        """分析性能扩展性"""
        print("\n📈 性能扩展性分析")
        print("=" * 50)
        
        if not self.latest_results:
            print("❌ 没有可用的benchmark数据")
            return
            
        results = self.latest_results['results']
        
        # 提取不同矩阵大小的性能数据
        cpu_performance = {}
        npu_performance = {}
        
        for key, data in results.items():
            if 'cpu_' in key:
                size = key.replace('cpu_', '')
                cpu_performance[int(size)] = data['gflops']
            elif 'pytorch_npu_' in key:
                size = key.replace('pytorch_npu_', '')
                npu_performance[int(size)] = data['gflops']
                
        # 分析扩展性
        print("📊 CPU性能扩展:")
        prev_size = None
        for size in sorted(cpu_performance.keys()):
            perf = cpu_performance[size]
            if prev_size:
                scale_factor = size / prev_size
                perf_factor = perf / cpu_performance[prev_size]
                efficiency = perf_factor / (scale_factor ** 3)  # 理论上应该是O(n^3)
                print(f"   {prev_size}x{prev_size} → {size}x{size}: {perf:.1f} GFLOPS (效率: {efficiency:.2f})")
            else:
                print(f"   {size}x{size}: {perf:.1f} GFLOPS")
            prev_size = size
            
        print("\n📊 NPU性能扩展:")
        prev_size = None
        for size in sorted(npu_performance.keys()):
            perf = npu_performance[size]
            if prev_size:
                scale_factor = size / prev_size
                perf_factor = perf / npu_performance[prev_size]
                efficiency = perf_factor / (scale_factor ** 3)
                print(f"   {prev_size}x{prev_size} → {size}x{size}: {perf:.1f} GFLOPS (效率: {efficiency:.2f})")
            else:
                print(f"   {size}x{size}: {perf:.1f} GFLOPS")
            prev_size = size
            
        return cpu_performance, npu_performance
        
    def analyze_npu_efficiency(self, npu_performance):
        """分析NPU效率"""
        print(f"\n🚀 NPU效率分析")
        print("=" * 50)
        
        # 昇腾910B2理论峰值性能
        theoretical_peak_fp32 = 50000  # GFLOPS (假设值)
        theoretical_peak_mixed = 100000  # GFLOPS (混合精度)
        
        print(f"🎯 理论峰值性能:")
        print(f"   FP32: {theoretical_peak_fp32:,} GFLOPS")
        print(f"   混合精度: {theoretical_peak_mixed:,} GFLOPS")
        
        print(f"\n📊 实际性能利用率:")
        for size in sorted(npu_performance.keys()):
            actual_perf = npu_performance[size]
            utilization_fp32 = (actual_perf / theoretical_peak_fp32) * 100
            utilization_mixed = (actual_perf / theoretical_peak_mixed) * 100
            
            print(f"   {size}x{size}: {actual_perf:.1f} GFLOPS")
            print(f"      vs FP32峰值: {utilization_fp32:.2f}%")
            print(f"      vs混合精度峰值: {utilization_mixed:.2f}%")
            
        # 分析性能瓶颈
        max_perf = max(npu_performance.values())
        max_size = max(npu_performance.keys())
        
        print(f"\n🔍 性能瓶颈分析:")
        print(f"   最大性能: {max_perf:.1f} GFLOPS ({max_size}x{max_size})")
        
        if max_perf < theoretical_peak_fp32 * 0.1:
            print(f"   🔴 瓶颈: 严重的算法或内存效率问题")
        elif max_perf < theoretical_peak_fp32 * 0.3:
            print(f"   🟡 瓶颈: 中等的优化空间，可能是内存带宽限制")
        else:
            print(f"   🟢 状态: 良好的硬件利用率")
            
    def project_boas_performance(self, npu_performance):
        """预测Boas性能潜力"""
        print(f"\n🎯 Boas性能潜力预测")
        print("=" * 50)
        
        # 获取Boas当前性能
        boas_current = 0.000011  # GFLOPS (2x2)
        
        # 基于PyTorch性能预测不同优化阶段的Boas性能
        optimization_stages = {
            "当前状态": {
                "factor": 1.0,
                "description": "2x2矩阵基础实现"
            },
            "修复编译器": {
                "factor": 6400,  # 16x16 vs 2x2
                "description": "支持16x16矩阵"
            },
            "算法优化": {
                "factor": 6400 * 16 * 2.5,  # 64x64 + 算法优化
                "description": "分块算法 + 循环优化"
            },
            "NPU并行": {
                "factor": 6400 * 16 * 2.5 * 16,  # + NPU并行
                "description": "多AI Core并行"
            },
            "高级优化": {
                "factor": 6400 * 16 * 2.5 * 16 * 3,  # + 高级优化
                "description": "混合精度 + kernel融合"
            }
        }
        
        print("📊 Boas性能路线图:")
        for stage, config in optimization_stages.items():
            projected_perf = boas_current * config['factor']
            
            # 与PyTorch性能对比
            if projected_perf > 1000:  # 只对大性能值进行对比
                pytorch_512 = npu_performance.get(512, 0)
                vs_pytorch = (projected_perf / pytorch_512) * 100 if pytorch_512 > 0 else 0
                print(f"   {stage:12}: {projected_perf:8.1f} GFLOPS ({vs_pytorch:5.1f}% vs PyTorch)")
            else:
                print(f"   {stage:12}: {projected_perf:8.6f} GFLOPS")
            print(f"                 {config['description']}")
            
        # 计算最终目标与PyTorch的竞争力
        final_projected = boas_current * optimization_stages["高级优化"]["factor"]
        pytorch_best = max(npu_performance.values())
        
        print(f"\n🏆 最终目标分析:")
        print(f"   Boas目标性能: {final_projected:.1f} GFLOPS")
        print(f"   PyTorch最佳: {pytorch_best:.1f} GFLOPS")
        
        if final_projected >= pytorch_best * 1.2:
            print(f"   🥇 结果: Boas有望超越PyTorch 20%+")
        elif final_projected >= pytorch_best:
            print(f"   🥈 结果: Boas有望达到PyTorch性能水平")
        elif final_projected >= pytorch_best * 0.8:
            print(f"   🥉 结果: Boas有望达到PyTorch 80%性能")
        else:
            print(f"   ⚠️ 结果: 需要更积极的优化策略")
            
        return optimization_stages, final_projected
        
    def analyze_optimization_priorities(self, cpu_performance, npu_performance):
        """分析优化优先级"""
        print(f"\n🎯 优化优先级分析")
        print("=" * 50)
        
        # 计算NPU vs CPU的加速比
        acceleration_ratios = {}
        for size in sorted(set(cpu_performance.keys()) & set(npu_performance.keys())):
            ratio = npu_performance[size] / cpu_performance[size]
            acceleration_ratios[size] = ratio
            
        print("📊 NPU加速比分析:")
        for size in sorted(acceleration_ratios.keys()):
            ratio = acceleration_ratios[size]
            print(f"   {size}x{size}: {ratio:.1f}x")
            
        # 识别性能跳跃点
        print(f"\n🔍 性能跳跃点分析:")
        npu_sizes = sorted(npu_performance.keys())
        for i in range(1, len(npu_sizes)):
            curr_size = npu_sizes[i]
            prev_size = npu_sizes[i-1]
            
            size_factor = curr_size / prev_size
            perf_factor = npu_performance[curr_size] / npu_performance[prev_size]
            
            if perf_factor > size_factor ** 2:  # 超线性加速
                print(f"   🚀 {prev_size}→{curr_size}: 超线性加速 ({perf_factor:.1f}x vs {size_factor**3:.1f}x期望)")
            elif perf_factor < size_factor:  # 次线性加速
                print(f"   ⚠️ {prev_size}→{curr_size}: 次线性加速 ({perf_factor:.1f}x vs {size_factor**3:.1f}x期望)")
                
        # 优化建议
        max_ratio = max(acceleration_ratios.values())
        min_ratio = min(acceleration_ratios.values())
        
        print(f"\n💡 优化建议:")
        if max_ratio > 8:
            print(f"   ✅ NPU已显示强劲加速能力 (最高{max_ratio:.1f}x)")
            print(f"   🎯 重点: 让Boas达到类似的NPU利用率")
        else:
            print(f"   ⚠️ NPU加速比有限 (最高{max_ratio:.1f}x)")
            print(f"   🎯 重点: 同时优化算法和NPU利用")
            
        if min_ratio < 3:
            print(f"   📊 小矩阵性能需要特别关注")
        if max_ratio / min_ratio > 3:
            print(f"   📈 大矩阵是NPU的强项，优先支持大规模计算")
            
    def create_performance_comparison_chart(self, cpu_performance, npu_performance):
        """创建性能对比图表"""
        print(f"\n📊 生成性能对比图表")
        print("=" * 50)
        
        # 准备数据
        sizes = sorted(set(cpu_performance.keys()) & set(npu_performance.keys()))
        cpu_values = [cpu_performance[size] for size in sizes]
        npu_values = [npu_performance[size] for size in sizes]
        
        # Boas投影数据 (基于理论计算)
        boas_projected = []
        base_boas = 0.000011
        for size in sizes:
            # 简化的投影：基于矩阵规模和预期优化效果
            if size <= 64:
                factor = (size/2)**3 * 1000  # 基础扩展
            elif size <= 256:
                factor = (size/2)**3 * 5000  # 算法优化
            else:
                factor = (size/2)**3 * 20000  # 全面优化
            projected = min(base_boas * factor, npu_performance[size] * 0.9)  # 不超过PyTorch的90%
            boas_projected.append(projected)
            
        # 创建图表数据
        chart_data = {
            'matrix_sizes': [f"{size}x{size}" for size in sizes],
            'cpu_performance': cpu_values,
            'pytorch_npu_performance': npu_values,
            'boas_projected_performance': boas_projected,
            'acceleration_ratios': [npu_values[i]/cpu_values[i] for i in range(len(sizes))]
        }
        
        # 保存图表数据
        os.makedirs('results/benchmarks', exist_ok=True)
        chart_file = f"results/benchmarks/performance_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(chart_file, 'w') as f:
            json.dump(chart_data, f, indent=2)
            
        print(f"📁 图表数据已保存: {chart_file}")
        
        # 打印表格形式的对比
        print(f"\n📋 性能对比表:")
        print(f"{'矩阵大小':<10} {'CPU(GFLOPS)':<15} {'PyTorch+NPU':<15} {'Boas预期':<15} {'NPU加速比':<10}")
        print("-" * 70)
        
        for i, size in enumerate(sizes):
            size_str = f"{size}x{size}"
            cpu_perf = cpu_values[i]
            npu_perf = npu_values[i]
            boas_perf = boas_projected[i]
            ratio = npu_perf / cpu_perf
            
            print(f"{size_str:<10} {cpu_perf:<15.1f} {npu_perf:<15.1f} {boas_perf:<15.1f} {ratio:<10.1f}")
            
        return chart_data
        
    def generate_comprehensive_report(self, cpu_performance, npu_performance, 
                                    optimization_stages, projected_performance):
        """生成综合分析报告"""
        print(f"\n📋 生成综合分析报告")
        print("=" * 50)
        
        report = {
            'analysis_date': datetime.now().isoformat(),
            'performance_data': {
                'cpu_performance': cpu_performance,
                'npu_performance': npu_performance,
                'max_cpu_gflops': max(cpu_performance.values()),
                'max_npu_gflops': max(npu_performance.values()),
                'max_acceleration_ratio': max(npu_performance[s]/cpu_performance[s] for s in cpu_performance.keys() if s in npu_performance)
            },
            'boas_projections': {
                'optimization_stages': optimization_stages,
                'final_projected_gflops': projected_performance,
                'competitiveness_vs_pytorch': (projected_performance / max(npu_performance.values())) * 100
            },
            'key_insights': [
                f"PyTorch NPU在{max(npu_performance, key=npu_performance.get)}x{max(npu_performance, key=npu_performance.get)}矩阵上达到峰值{max(npu_performance.values()):.1f} GFLOPS",
                f"NPU相对CPU的最大加速比为{max(npu_performance[s]/cpu_performance[s] for s in cpu_performance.keys() if s in npu_performance):.1f}x",
                f"Boas有潜力达到{projected_performance:.1f} GFLOPS ({(projected_performance/max(npu_performance.values()))*100:.1f}% vs PyTorch)"
            ],
            'recommendations': [
                "优先实现大矩阵支持以发挥NPU优势",
                "重点优化内存访问模式和算法效率", 
                "采用混合精度和并行策略",
                "建立持续性能监控和优化机制"
            ]
        }
        
        # 保存报告
        report_file = f"results/benchmarks/benchmark_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        print(f"📁 综合分析报告已保存: {report_file}")
        
        # 创建Markdown摘要
        md_file = report_file.replace('.json', '.md')
        with open(md_file, 'w') as f:
            f.write(f"""# 📊 Benchmark性能对比分析报告

## 🎯 分析摘要
- **分析日期**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **CPU最佳性能**: {report['performance_data']['max_cpu_gflops']:.1f} GFLOPS
- **NPU最佳性能**: {report['performance_data']['max_npu_gflops']:.1f} GFLOPS
- **最大加速比**: {report['performance_data']['max_acceleration_ratio']:.1f}x

## 🚀 Boas性能预测
- **最终目标**: {report['boas_projections']['final_projected_gflops']:.1f} GFLOPS
- **竞争力**: {report['boas_projections']['competitiveness_vs_pytorch']:.1f}% vs PyTorch

## 💡 关键洞察
""")
            for insight in report['key_insights']:
                f.write(f"- {insight}\n")
                
            f.write(f"\n## 📋 优化建议\n")
            for rec in report['recommendations']:
                f.write(f"- {rec}\n")
                
        print(f"📄 分析摘要已保存: {md_file}")
        
        return report

def main():
    """主分析流程"""
    print("📊 Benchmark性能对比分析器")
    print("=" * 60)
    
    analyzer = BenchmarkAnalyzer()
    
    if not analyzer.latest_results:
        print("❌ 无法继续分析，请先运行benchmark测试")
        return
        
    # 1. 性能扩展性分析
    cpu_perf, npu_perf = analyzer.analyze_performance_scaling()
    
    # 2. NPU效率分析
    analyzer.analyze_npu_efficiency(npu_perf)
    
    # 3. Boas性能预测
    opt_stages, projected_perf = analyzer.project_boas_performance(npu_perf)
    
    # 4. 优化优先级分析
    analyzer.analyze_optimization_priorities(cpu_perf, npu_perf)
    
    # 5. 性能对比图表
    chart_data = analyzer.create_performance_comparison_chart(cpu_perf, npu_perf)
    
    # 6. 综合报告
    report = analyzer.generate_comprehensive_report(cpu_perf, npu_perf, opt_stages, projected_perf)
    
    print(f"\n🎉 Benchmark分析完成!")
    print(f"📊 关键发现: Boas有潜力达到{projected_perf:.1f} GFLOPS")
    print(f"🏆 竞争力: {(projected_perf/max(npu_perf.values()))*100:.1f}% vs PyTorch最佳性能")

if __name__ == "__main__":
    main()
