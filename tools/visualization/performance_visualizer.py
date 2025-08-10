#!/usr/bin/env python3
"""
📊 性能对比可视化工具
生成直观的性能对比图表和分析报告
"""

import json
import os
from datetime import datetime

class PerformanceVisualizer:
    def __init__(self):
        self.load_latest_data()
        
    def load_latest_data(self):
        """加载最新的性能数据"""
        benchmark_dir = "results/benchmarks"
        if not os.path.exists(benchmark_dir):
            print("❌ 未找到benchmark结果目录")
            return
            
        # 查找最新的性能对比数据
        comparison_files = [f for f in os.listdir(benchmark_dir) if f.startswith('performance_comparison_')]
        if not comparison_files:
            print("❌ 未找到性能对比数据")
            return
            
        latest_file = max(comparison_files)
        file_path = os.path.join(benchmark_dir, latest_file)
        
        try:
            with open(file_path, 'r') as f:
                self.data = json.load(f)
            print(f"✅ 加载了性能对比数据: {latest_file}")
        except Exception as e:
            print(f"❌ 加载数据失败: {e}")
            self.data = None
            
    def create_ascii_charts(self):
        """创建ASCII图表"""
        if not self.data:
            return
            
        print("\n📊 性能对比可视化")
        print("=" * 60)
        
        sizes = self.data['matrix_sizes']
        cpu_perf = self.data['cpu_performance']
        npu_perf = self.data['pytorch_npu_performance']
        boas_proj = self.data['boas_projected_performance']
        
        # 1. 性能对比条形图
        print("\n📈 性能对比 (GFLOPS)")
        print("-" * 60)
        
        max_perf = max(max(cpu_perf), max(npu_perf), max(boas_proj))
        
        for i, size in enumerate(sizes):
            cpu_val = cpu_perf[i]
            npu_val = npu_perf[i]
            boas_val = boas_proj[i]
            
            # 计算条形长度 (最大40字符)
            cpu_bar = int((cpu_val / max_perf) * 40)
            npu_bar = int((npu_val / max_perf) * 40)
            boas_bar = int((boas_val / max_perf) * 40)
            
            print(f"{size:8}")
            print(f"  CPU    {'█' * cpu_bar:<40} {cpu_val:8.1f}")
            print(f"  PyTorch{'█' * npu_bar:<40} {npu_val:8.1f}")
            print(f"  Boas   {'█' * boas_bar:<40} {boas_val:8.1f}")
            print()
            
        # 2. 加速比图表
        print("\n🚀 NPU加速比 (vs CPU)")
        print("-" * 40)
        
        ratios = self.data['acceleration_ratios']
        max_ratio = max(ratios)
        
        for i, size in enumerate(sizes):
            ratio = ratios[i]
            bar_len = int((ratio / max_ratio) * 30)
            
            if ratio >= 5:
                icon = "🚀"
            elif ratio >= 2:
                icon = "⚡"
            elif ratio >= 1:
                icon = "📈"
            else:
                icon = "📉"
                
            print(f"{size:8} {icon} {'█' * bar_len:<30} {ratio:5.1f}x")
            
    def analyze_performance_trends(self):
        """分析性能趋势"""
        print(f"\n📊 性能趋势分析")
        print("=" * 60)
        
        if not self.data:
            return
            
        cpu_perf = self.data['cpu_performance']
        npu_perf = self.data['pytorch_npu_performance']
        boas_proj = self.data['boas_projected_performance']
        ratios = self.data['acceleration_ratios']
        
        # 分析CPU性能趋势
        print("🖥️ CPU性能趋势:")
        cpu_growth = []
        for i in range(1, len(cpu_perf)):
            growth = cpu_perf[i] / cpu_perf[i-1]
            cpu_growth.append(growth)
            print(f"   📈 增长: {growth:.1f}x")
            
        avg_cpu_growth = sum(cpu_growth) / len(cpu_growth)
        print(f"   📊 平均增长: {avg_cpu_growth:.1f}x")
        
        # 分析NPU性能趋势
        print(f"\n🚀 NPU性能趋势:")
        npu_growth = []
        for i in range(1, len(npu_perf)):
            growth = npu_perf[i] / npu_perf[i-1]
            npu_growth.append(growth)
            print(f"   📈 增长: {growth:.1f}x")
            
        avg_npu_growth = sum(npu_growth) / len(npu_growth)
        print(f"   📊 平均增长: {avg_npu_growth:.1f}x")
        
        # NPU vs CPU趋势
        print(f"\n⚡ 加速比趋势:")
        ratio_trend = "上升" if ratios[-1] > ratios[0] else "下降"
        print(f"   📊 总体趋势: {ratio_trend}")
        print(f"   🎯 最佳加速比: {max(ratios):.1f}x ({self.data['matrix_sizes'][ratios.index(max(ratios))]})")
        
        # Boas竞争力分析
        print(f"\n🎯 Boas竞争力分析:")
        for i, size in enumerate(self.data['matrix_sizes']):
            boas_vs_npu = (boas_proj[i] / npu_perf[i]) * 100
            boas_vs_cpu = (boas_proj[i] / cpu_perf[i]) * 100
            
            print(f"   {size}: vs PyTorch {boas_vs_npu:5.1f}%, vs CPU {boas_vs_cpu:5.1f}%")
            
    def identify_key_insights(self):
        """识别关键洞察"""
        print(f"\n💡 关键洞察")
        print("=" * 60)
        
        if not self.data:
            return
            
        cpu_perf = self.data['cpu_performance']
        npu_perf = self.data['pytorch_npu_performance']
        boas_proj = self.data['boas_projected_performance']
        ratios = self.data['acceleration_ratios']
        
        insights = []
        
        # 1. NPU性能突破点
        for i in range(len(ratios)):
            if ratios[i] > 5:
                insights.append(f"🚀 NPU在{self.data['matrix_sizes'][i]}矩阵上实现{ratios[i]:.1f}x显著加速")
                break
                
        # 2. 最佳性能规模
        max_npu_idx = npu_perf.index(max(npu_perf))
        insights.append(f"🏆 NPU峰值性能出现在{self.data['matrix_sizes'][max_npu_idx]} ({max(npu_perf):.1f} GFLOPS)")
        
        # 3. Boas机会点
        max_boas_vs_npu = max((boas_proj[i] / npu_perf[i]) * 100 for i in range(len(boas_proj)))
        insights.append(f"🎯 Boas最有竞争力可达PyTorch的{max_boas_vs_npu:.1f}%")
        
        # 4. 优化机会
        if max(ratios) > 5:
            insights.append(f"✅ NPU优化空间巨大，Boas应重点发挥大矩阵优势")
        else:
            insights.append(f"⚠️ NPU加速有限，Boas需要创新算法优化")
            
        # 5. 规模效应
        if npu_perf[-1] / npu_perf[0] > 100:
            insights.append(f"📈 大规模矩阵是NPU的强项，性能提升{npu_perf[-1]/npu_perf[0]:.0f}倍")
            
        for insight in insights:
            print(f"   {insight}")
            
        return insights
        
    def create_performance_matrix(self):
        """创建性能矩阵表"""
        print(f"\n📋 详细性能对比矩阵")
        print("=" * 80)
        
        if not self.data:
            return
            
        # 表头
        print(f"{'矩阵规模':<12} {'CPU':<12} {'PyTorch+NPU':<15} {'Boas预期':<12} {'加速比':<10} {'Boas竞争力':<12}")
        print("-" * 80)
        
        cpu_perf = self.data['cpu_performance']
        npu_perf = self.data['pytorch_npu_performance']
        boas_proj = self.data['boas_projected_performance']
        ratios = self.data['acceleration_ratios']
        
        for i, size in enumerate(self.data['matrix_sizes']):
            cpu_val = cpu_perf[i]
            npu_val = npu_perf[i]
            boas_val = boas_proj[i]
            ratio = ratios[i]
            competitiveness = (boas_val / npu_val) * 100
            
            print(f"{size:<12} {cpu_val:<12.1f} {npu_val:<15.1f} {boas_val:<12.1f} {ratio:<10.1f} {competitiveness:<12.1f}%")
            
    def generate_executive_summary(self):
        """生成执行摘要"""
        print(f"\n📋 执行摘要")
        print("=" * 60)
        
        if not self.data:
            return
            
        cpu_perf = self.data['cpu_performance']
        npu_perf = self.data['pytorch_npu_performance']
        boas_proj = self.data['boas_projected_performance']
        ratios = self.data['acceleration_ratios']
        
        # 关键指标
        max_cpu = max(cpu_perf)
        max_npu = max(npu_perf)
        max_boas = max(boas_proj)
        max_ratio = max(ratios)
        
        # 竞争力评估
        avg_competitiveness = sum((boas_proj[i] / npu_perf[i]) * 100 for i in range(len(boas_proj))) / len(boas_proj)
        
        summary = {
            "performance_leadership": "PyTorch+NPU" if max_npu > max_cpu * 2 else "CPU",
            "npu_acceleration": f"{max_ratio:.1f}x",
            "boas_potential": f"{max_boas:.1f} GFLOPS",
            "competitiveness": f"{avg_competitiveness:.1f}%",
            "recommendation": ""
        }
        
        # 生成建议
        if avg_competitiveness >= 80:
            summary["recommendation"] = "🏆 Boas具备强竞争力，建议加速实施"
        elif avg_competitiveness >= 60:
            summary["recommendation"] = "✅ Boas有良好潜力，需要重点优化"
        elif avg_competitiveness >= 40:
            summary["recommendation"] = "⚠️ Boas需要创新突破，重新设计算法"
        else:
            summary["recommendation"] = "🔴 当前策略需要根本性调整"
            
        print(f"🎯 性能领导者: {summary['performance_leadership']}")
        print(f"🚀 最大NPU加速: {summary['npu_acceleration']}")
        print(f"💪 Boas潜力: {summary['boas_potential']}")
        print(f"🏁 平均竞争力: {summary['competitiveness']}")
        print(f"💡 策略建议: {summary['recommendation']}")
        
        return summary
        
    def save_visualization_report(self, insights, summary):
        """保存可视化报告"""
        print(f"\n📁 保存可视化报告")
        print("=" * 60)
        
        report = {
            'visualization_date': datetime.now().isoformat(),
            'performance_data': self.data,
            'key_insights': insights,
            'executive_summary': summary,
            'visualization_notes': [
                "所有性能数据基于实际benchmark测试",
                "Boas预期性能基于优化潜力分析",
                "NPU加速比显示了硬件优势",
                "大矩阵规模是NPU的性能甜点"
            ]
        }
        
        # 保存JSON报告
        os.makedirs('results/visualization', exist_ok=True)
        report_file = f"results/visualization/performance_visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        print(f"📊 可视化报告已保存: {report_file}")
        
        # 创建Markdown总结
        md_file = report_file.replace('.json', '.md')
        with open(md_file, 'w') as f:
            f.write(f"""# 📊 Boas vs PyTorch vs CPU 性能对比可视化报告

## 🎯 执行摘要
- **性能领导者**: {summary['performance_leadership']}
- **最大NPU加速**: {summary['npu_acceleration']}
- **Boas潜力**: {summary['boas_potential']}
- **平均竞争力**: {summary['competitiveness']}
- **策略建议**: {summary['recommendation']}

## 💡 关键洞察
""")
            for insight in insights:
                f.write(f"- {insight}\n")
                
            f.write(f"\n## 📊 性能数据\n")
            f.write(f"| 矩阵规模 | CPU (GFLOPS) | PyTorch+NPU | Boas预期 | 加速比 |\n")
            f.write(f"|----------|--------------|-------------|----------|--------|\n")
            
            for i, size in enumerate(self.data['matrix_sizes']):
                cpu_val = self.data['cpu_performance'][i]
                npu_val = self.data['pytorch_npu_performance'][i]
                boas_val = self.data['boas_projected_performance'][i]
                ratio = self.data['acceleration_ratios'][i]
                f.write(f"| {size} | {cpu_val:.1f} | {npu_val:.1f} | {boas_val:.1f} | {ratio:.1f}x |\n")
                
        print(f"📄 可视化摘要已保存: {md_file}")
        
        return report_file

def main():
    """主可视化流程"""
    print("📊 性能对比可视化工具")
    print("=" * 60)
    
    visualizer = PerformanceVisualizer()
    
    if not visualizer.data:
        print("❌ 无法生成可视化，请先运行benchmark测试")
        return
        
    # 1. 创建ASCII图表
    visualizer.create_ascii_charts()
    
    # 2. 性能趋势分析
    visualizer.analyze_performance_trends()
    
    # 3. 关键洞察
    insights = visualizer.identify_key_insights()
    
    # 4. 性能矩阵
    visualizer.create_performance_matrix()
    
    # 5. 执行摘要
    summary = visualizer.generate_executive_summary()
    
    # 6. 保存报告
    report_file = visualizer.save_visualization_report(insights, summary)
    
    print(f"\n🎉 可视化分析完成!")
    print(f"📁 查看详细报告: {report_file}")

if __name__ == "__main__":
    main()
