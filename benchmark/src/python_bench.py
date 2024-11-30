import numpy as np
import time
import psutil
import pandas as pd
import os

def benchmark_numpy(size):
    # 创建随机矩阵
    A = np.random.rand(size, size).astype(np.float32)
    B = np.random.rand(size, size).astype(np.float32)
    
    # 记录开始状态
    start_time = time.time()
    process = psutil.Process()
    start_memory = process.memory_info().rss
    
    # 执行矩阵乘法
    C = np.matmul(A, B)
    
    # 记录结束状态
    end_time = time.time()
    end_memory = process.memory_info().rss
    
    # 计算结果
    time_ms = (end_time - start_time) * 1000
    memory_kb = (end_memory - start_memory) / 1024
    
    # 保存结果
    with open('../results/results.csv', 'a', encoding='utf-8') as f:
        f.write(f"matrix_multiplication,Python,{size},{time_ms}\n")

if __name__ == '__main__':
    sizes = [64, 128, 256, 512, 1024]
    for size in sizes:
        benchmark_numpy(size)