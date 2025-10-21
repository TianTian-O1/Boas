#!/usr/bin/env python3
"""
BOAS GPU vs NPU 性能对比基准测试
比较Nvidia GPU和华为昇腾NPU的矩阵乘法性能
"""

import subprocess
import time
import sys
import os
from typing import List, Dict, Tuple

class BenchmarkRunner:
    def __init__(self, compiler_path: str = "./build/matrix-compiler"):
        self.compiler_path = compiler_path
        self.results = []

    def create_test_file(self, size: int, device: str) -> str:
        """创建测试文件"""
        filename = f"test_matmul_{device}_{size}.bs"

        code = f"""import tensor

def main():
    print("测试 {size}x{size} 矩阵乘法 on {device}")

    # 创建随机矩阵
    A = tensor.random({size}, {size})
    B = tensor.random({size}, {size})

    # 执行矩阵乘法
    C = tensor.matmul(A, B)

    print("完成 {size}x{size} 矩阵乘法")

if __name__ == "__main__":
    main()
"""
        with open(filename, 'w') as f:
            f.write(code)

        return filename

    def compile_and_run(self, test_file: str, device: str) -> Tuple[bool, float]:
        """编译并运行测试"""
        output_bin = test_file.replace('.bs', '')

        # 编译
        compile_cmd = [
            self.compiler_path,
            test_file,
            '-o', output_bin,
            f'--device={device}'
        ]

        try:
            print(f"  编译: {' '.join(compile_cmd)}")
            result = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                print(f"  编译失败: {result.stderr}")
                return False, 0.0

            # 运行并计时
            print(f"  运行: {output_bin}")
            start = time.time()
            result = subprocess.run([f'./{output_bin}'], capture_output=True, text=True, timeout=120)
            end = time.time()

            if result.returncode != 0:
                print(f"  运行失败: {result.stderr}")
                return False, 0.0

            elapsed = end - start
            print(f"  耗时: {elapsed:.4f}s")

            # 清理
            os.remove(output_bin)

            return True, elapsed

        except subprocess.TimeoutExpired:
            print(f"  超时")
            return False, 0.0
        except Exception as e:
            print(f"  错误: {e}")
            return False, 0.0
        finally:
            if os.path.exists(test_file):
                os.remove(test_file)

    def run_benchmark(self, sizes: List[int], devices: List[str], iterations: int = 3):
        """运行基准测试"""
        print("\n" + "="*60)
        print("BOAS GPU vs NPU 性能对比基准测试")
        print("="*60 + "\n")

        for size in sizes:
            print(f"\n矩阵大小: {size}x{size}")
            print("-" * 60)

            for device in devices:
                print(f"\n  设备: {device}")

                times = []
                for i in range(iterations):
                    print(f"    迭代 {i+1}/{iterations}...")
                    test_file = self.create_test_file(size, device)

                    success, elapsed = self.compile_and_run(test_file, device)
                    if success:
                        times.append(elapsed)
                    else:
                        print(f"    跳过")
                        break

                if times:
                    avg_time = sum(times) / len(times)
                    min_time = min(times)
                    max_time = max(times)

                    # 计算GFLOPS (2*M*N*K operations)
                    ops = 2 * size * size * size
                    gflops = ops / avg_time / 1e9

                    result = {
                        'size': size,
                        'device': device,
                        'avg_time': avg_time,
                        'min_time': min_time,
                        'max_time': max_time,
                        'gflops': gflops
                    }
                    self.results.append(result)

                    print(f"\n    结果:")
                    print(f"      平均耗时: {avg_time:.4f}s")
                    print(f"      最小耗时: {min_time:.4f}s")
                    print(f"      最大耗时: {max_time:.4f}s")
                    print(f"      性能: {gflops:.2f} GFLOPS")

    def print_summary(self):
        """打印汇总结果"""
        print("\n" + "="*60)
        print("基准测试汇总")
        print("="*60 + "\n")

        # 按矩阵大小分组
        sizes = sorted(set(r['size'] for r in self.results))

        print(f"{'大小':<12} {'设备':<10} {'平均耗时(s)':<15} {'GFLOPS':<12} {'相对性能':<12}")
        print("-" * 70)

        for size in sizes:
            size_results = [r for r in self.results if r['size'] == size]

            # 找到最快的作为基准
            fastest = min(size_results, key=lambda x: x['avg_time'])

            for r in size_results:
                relative = fastest['avg_time'] / r['avg_time']
                speedup_str = f"{relative:.2f}x"

                print(f"{r['size']:<12} {r['device']:<10} {r['avg_time']:<15.4f} "
                      f"{r['gflops']:<12.2f} {speedup_str:<12}")

            print()

    def save_results(self, filename: str = "gpu_npu_benchmark_results.csv"):
        """保存结果到CSV"""
        import csv

        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['size', 'device', 'avg_time', 'min_time', 'max_time', 'gflops'])
            writer.writeheader()
            writer.writerows(self.results)

        print(f"\n结果已保存到: {filename}")

def main():
    # 配置
    compiler_path = "./build/matrix-compiler"
    sizes = [128, 256, 512, 1024, 2048, 4096]
    devices = ['gpu', 'npu', 'cpu']  # 可选设备
    iterations = 3

    # 检查编译器是否存在
    if not os.path.exists(compiler_path):
        print(f"错误: 找不到编译器 {compiler_path}")
        print("请先编译项目: mkdir build && cd build && cmake .. && make")
        sys.exit(1)

    # 运行基准测试
    runner = BenchmarkRunner(compiler_path)

    # 只测试可用的设备
    available_devices = []

    # 检测CUDA
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, timeout=5)
        if result.returncode == 0:
            available_devices.append('gpu')
            print("✓ 检测到 Nvidia GPU")
    except:
        print("✗ 未检测到 Nvidia GPU")

    # 检测NPU
    try:
        result = subprocess.run(['npu-smi', 'info'], capture_output=True, timeout=5)
        if result.returncode == 0:
            available_devices.append('npu')
            print("✓ 检测到 Ascend NPU")
    except:
        print("✗ 未检测到 Ascend NPU")

    # CPU总是可用
    available_devices.append('cpu')

    if len(available_devices) == 1:
        print("\n警告: 只有CPU可用，无法进行对比测试")

    print(f"\n可用设备: {', '.join(available_devices)}\n")

    runner.run_benchmark(sizes, available_devices, iterations)
    runner.print_summary()
    runner.save_results()

if __name__ == "__main__":
    main()
