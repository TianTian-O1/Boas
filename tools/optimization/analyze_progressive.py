#!/usr/bin/env python3
"""渐进式测试结果分析"""
import csv
import os

def analyze_progressive_results():
    if not os.path.exists("progressive_results.csv"):
        print("❌ 未找到测试结果文件")
        return
        
    print("📊 渐进式测试结果分析")
    print("=" * 40)
    
    with open("progressive_results.csv", 'r') as f:
        reader = csv.reader(f)
        results = list(reader)
        
    if not results:
        print("❌ 结果文件为空")
        return
        
    print("Size     | Time(s)  | GFLOPS   | 相对2x2")
    print("-" * 40)
    
    base_perf = 0.000011  # 2x2基准性能
    
    for row in results:
        try:
            size, time_s, gflops = row
            gflops_val = float(gflops)
            improvement = gflops_val / base_perf
            print(f"{size:8} | {time_s:8} | {gflops:8} | {improvement:.1f}x")
        except:
            continue
            
    print("")
    print("🎯 分析结论:")
    if len(results) > 0:
        last_gflops = float(results[-1][2])
        if last_gflops > 0.01:
            print(f"✅ 大矩阵性能显著提升")
        elif last_gflops > 0.001:
            print(f"⚡ 性能有所改善，需进一步优化")
        else:
            print(f"⚠️ 性能提升有限，需要算法优化")

if __name__ == "__main__":
    analyze_progressive_results()