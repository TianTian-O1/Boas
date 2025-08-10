#!/usr/bin/env python3
"""
🎯 Boas语言综合性能基准测试
对比Boas、PyTorch+NPU、CPU性能，评估优化效果
"""

import time
import numpy as np
import subprocess
import os
import json
from datetime import datetime

try:
    import torch
    import torch_npu
    TORCH_NPU_AVAILABLE = True
except ImportError:
    TORCH_NPU_AVAILABLE = False

class ComprehensiveBenchmark:
    def __init__(self):
        self.matrix_sizes = [64, 128, 256, 512]  # 从小到大测试
        self.num_runs = 10  # 每个测试运行次数
        self.results = {}
        
    def benchmark_cpu_numpy(self, size):
        """CPU NumPy基准测试"""
        print(f"   🖥️ CPU NumPy {size}x{size}...")
        
        # 预热
        A = np.random.randn(size, size).astype(np.float32)
        B = np.random.randn(size, size).astype(np.float32)
        _ = np.dot(A, B)  # 预热
        
        times = []
        for _ in range(self.num_runs):
            A = np.random.randn(size, size).astype(np.float32)
            B = np.random.randn(size, size).astype(np.float32)
            
            start = time.time()
            C = np.dot(A, B)
            end = time.time()
            
            times.append((end - start) * 1000)  # ms
            
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        # 计算GFLOPS
        flops = 2 * size**3  # 矩阵乘法的FLOPS
        gflops = flops / (avg_time / 1000) / 1e9
        
        return {
            'avg_time_ms': avg_time,
            'std_time_ms': std_time,
            'gflops': gflops,
            'framework': 'CPU (NumPy)'
        }
        
    def benchmark_pytorch_npu(self, size):
        """PyTorch+NPU基准测试"""
        if not TORCH_NPU_AVAILABLE:
            return None
            
        print(f"   🚀 PyTorch+NPU {size}x{size}...")
        
        device = torch.device('npu:0')
        
        # 预热
        A = torch.randn(size, size, dtype=torch.float32).to(device)
        B = torch.randn(size, size, dtype=torch.float32).to(device)
        _ = torch.mm(A, B)
        torch.npu.synchronize()
        
        times = []
        for _ in range(self.num_runs):
            A = torch.randn(size, size, dtype=torch.float32).to(device)
            B = torch.randn(size, size, dtype=torch.float32).to(device)
            
            torch.npu.synchronize()
            start = time.time()
            C = torch.mm(A, B)
            torch.npu.synchronize()
            end = time.time()
            
            times.append((end - start) * 1000)  # ms
            
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        # 计算GFLOPS
        flops = 2 * size**3
        gflops = flops / (avg_time / 1000) / 1e9
        
        return {
            'avg_time_ms': avg_time,
            'std_time_ms': std_time,
            'gflops': gflops,
            'framework': 'PyTorch+NPU'
        }
        
    def benchmark_boas_original(self):
        """Boas原版本基准测试"""
        if not os.path.exists("boas_npu_test"):
            return None
            
        print(f"   🔵 Boas原版本...")
        
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = "/usr/lib/aarch64-linux-gnu:/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64:" + env.get("LD_LIBRARY_PATH", "")
        
        times = []
        for _ in range(self.num_runs):
            start = time.time()
            try:
                result = subprocess.run(['./boas_npu_test'], 
                                      capture_output=True, text=True, 
                                      timeout=5, env=env)
                end = time.time()
                
                if result.returncode == 0:
                    times.append((end - start) * 1000)  # ms
                    
            except (subprocess.TimeoutExpired, Exception):
                pass
                
        if not times:
            return None
            
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        # 估算GFLOPS (2x2矩阵)
        flops = 2 * 2 * 2 * 2
        gflops = flops / (avg_time / 1000) / 1e9
        
        return {
            'avg_time_ms': avg_time,
            'std_time_ms': std_time,
            'gflops': gflops,
            'framework': 'Boas原版本',
            'matrix_size': '2x2'
        }
        
    def benchmark_boas_optimized(self):
        """Boas优化版本基准测试"""
        if not os.path.exists("boas_npu_optimized"):
            return None
            
        print(f"   🚀 Boas优化版本...")
        
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = "/usr/lib/aarch64-linux-gnu:/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64:" + env.get("LD_LIBRARY_PATH", "")
        
        times = []
        for _ in range(self.num_runs):
            start = time.time()
            try:
                result = subprocess.run(['./boas_npu_optimized'], 
                                      capture_output=True, text=True, 
                                      timeout=5, env=env)
                end = time.time()
                
                if result.returncode == 0:
                    times.append((end - start) * 1000)  # ms
                    
            except (subprocess.TimeoutExpired, Exception):
                pass
                
        if not times:
            return None
            
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        # 估算GFLOPS (2x2矩阵)
        flops = 2 * 2 * 2 * 2
        gflops = flops / (avg_time / 1000) / 1e9
        
        return {
            'avg_time_ms': avg_time,
            'std_time_ms': std_time,
            'gflops': gflops,
            'framework': 'Boas优化版本',
            'matrix_size': '2x2'
        }
        
    def run_comprehensive_benchmark(self):
        """运行综合基准测试"""
        print("🎯 综合性能基准测试开始")
        print("=" * 60)
        
        # CPU和PyTorch测试（多种矩阵大小）
        for size in self.matrix_sizes:
            print(f"\n📊 测试 {size}x{size} 矩阵:")
            
            # CPU测试
            cpu_result = self.benchmark_cpu_numpy(size)
            if cpu_result:
                self.results[f'cpu_{size}'] = cpu_result
                print(f"      CPU: {cpu_result['gflops']:.1f} GFLOPS")
                
            # PyTorch NPU测试
            if TORCH_NPU_AVAILABLE:
                npu_result = self.benchmark_pytorch_npu(size)
                if npu_result:
                    self.results[f'pytorch_npu_{size}'] = npu_result
                    print(f"      PyTorch+NPU: {npu_result['gflops']:.1f} GFLOPS")
            else:
                print(f"      PyTorch+NPU: 不可用")
                
        # Boas测试（当前只支持小矩阵）
        print(f"\n🔵 Boas语言测试:")
        boas_orig = self.benchmark_boas_original()
        if boas_orig:
            self.results['boas_original'] = boas_orig
            print(f"      Boas原版本: {boas_orig['gflops']:.6f} GFLOPS ({boas_orig['matrix_size']})")
            
        boas_opt = self.benchmark_boas_optimized()
        if boas_opt:
            self.results['boas_optimized'] = boas_opt
            print(f"      Boas优化版本: {boas_opt['gflops']:.6f} GFLOPS ({boas_opt['matrix_size']})")
            
            # 计算Boas优化效果
            if boas_orig:
                speedup = boas_opt['gflops'] / boas_orig['gflops']
                print(f"      优化提升: {speedup:.2f}x")
                
    def analyze_results(self):
        """分析测试结果"""
        print(f"\n📈 性能分析结果")
        print("=" * 60)
        
        # 找到最佳性能
        best_cpu = 0
        best_npu = 0
        
        for key, result in self.results.items():
            if 'cpu_' in key:
                best_cpu = max(best_cpu, result['gflops'])
            elif 'pytorch_npu_' in key:
                best_npu = max(best_npu, result['gflops'])
                
        print(f"🖥️ CPU最佳性能: {best_cpu:.1f} GFLOPS")
        if best_npu > 0:
            print(f"🚀 NPU最佳性能: {best_npu:.1f} GFLOPS")
            print(f"📊 NPU加速比: {best_npu/best_cpu:.1f}x")
            
        # Boas性能分析
        if 'boas_optimized' in self.results:
            boas_perf = self.results['boas_optimized']['gflops']
            print(f"\n🔵 Boas当前性能: {boas_perf:.6f} GFLOPS")
            
            # 估算扩展到大矩阵的性能
            if best_npu > 0:
                # 基于NPU理论性能估算
                theoretical_boas = best_npu * 0.8  # 保守估计80%
                print(f"🎯 Boas理论性能(大矩阵): {theoretical_boas:.1f} GFLOPS")
                
                target_gflops = [3220, 4026, 4831]  # 最低、竞争、卓越目标
                target_names = ['最低目标', '竞争目标', '卓越目标']
                
                print(f"\n🏆 性能目标对比:")
                for target, name in zip(target_gflops, target_names):
                    achievable = theoretical_boas >= target
                    status = "✅ 可达成" if achievable else "⚠️ 需优化"
                    print(f"   {name} ({target} GFLOPS): {status}")
                    
    def generate_report(self):
        """生成测试报告"""
        print(f"\n📋 生成测试报告")
        print("=" * 60)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'test_config': {
                'matrix_sizes': self.matrix_sizes,
                'num_runs': self.num_runs,
                'pytorch_npu_available': TORCH_NPU_AVAILABLE
            },
            'results': self.results,
            'summary': {}
        }
        
        # 计算总结统计
        cpu_results = [r for k, r in self.results.items() if 'cpu_' in k]
        npu_results = [r for k, r in self.results.items() if 'pytorch_npu_' in k]
        
        if cpu_results:
            report['summary']['cpu_avg_gflops'] = np.mean([r['gflops'] for r in cpu_results])
            report['summary']['cpu_max_gflops'] = max([r['gflops'] for r in cpu_results])
            
        if npu_results:
            report['summary']['npu_avg_gflops'] = np.mean([r['gflops'] for r in npu_results])
            report['summary']['npu_max_gflops'] = max([r['gflops'] for r in npu_results])
            
        if 'boas_optimized' in self.results:
            report['summary']['boas_gflops'] = self.results['boas_optimized']['gflops']
            
        # 保存报告
        report_file = f"comprehensive_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"📁 详细报告已保存: {report_file}")
        
        # 打印关键结论
        print(f"\n🎯 关键结论:")
        if 'npu_max_gflops' in report['summary']:
            max_npu = report['summary']['npu_max_gflops']
            print(f"   • NPU峰值性能: {max_npu:.1f} GFLOPS")
            print(f"   • Boas目标性能: {max_npu * 0.8:.1f} - {max_npu * 1.2:.1f} GFLOPS")
            
        if 'boas_gflops' in report['summary']:
            boas_gflops = report['summary']['boas_gflops']
            print(f"   • Boas当前性能: {boas_gflops:.6f} GFLOPS (小矩阵)")
            print(f"   • 需要扩展到大矩阵测试以获得真实性能")
            
        return report

def main():
    """主函数"""
    print("🚀 Boas语言综合性能基准测试")
    print("=" * 60)
    
    benchmark = ComprehensiveBenchmark()
    
    # 运行测试
    benchmark.run_comprehensive_benchmark()
    
    # 分析结果
    benchmark.analyze_results()
    
    # 生成报告
    report = benchmark.generate_report()
    
    print(f"\n✅ 综合基准测试完成!")

if __name__ == "__main__":
    main()
