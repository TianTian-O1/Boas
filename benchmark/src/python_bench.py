import numpy as np
import time
import psutil
import pandas as pd
import os
import gc
from array import array
from itertools import repeat
from functools import partial

def get_memory_usage():
    """获取当前进程的内存使用"""
    process = psutil.Process()
    return process.memory_info().rss / 1024  # 转换为KB

def pure_python_matmul_optimized(A, B):
    """优化的纯 Python 矩阵乘法实现"""
    n = len(A)
    # 使用 array 模块存储浮点数，比列表更高效
    result = [array('f', repeat(0.0, n)) for _ in range(n)]
    
    # 预计算 B 的转置，提高缓存命中率
    B_trans = [array('f', (B[j][i] for j in range(n))) for i in range(n)]
    
    # 分块大小
    BLOCK_SIZE = 32
    
    # 分块矩阵乘法
    for i0 in range(0, n, BLOCK_SIZE):
        for j0 in range(0, n, BLOCK_SIZE):
            for k0 in range(0, n, BLOCK_SIZE):
                # 计算块的实际大小
                i_end = min(i0 + BLOCK_SIZE, n)
                j_end = min(j0 + BLOCK_SIZE, n)
                k_end = min(k0 + BLOCK_SIZE, n)
                
                # 计算当前块
                for i in range(i0, i_end):
                    row_i = A[i]
                    res_i = result[i]
                    for j in range(j0, j_end):
                        b_col = B_trans[j]
                        # 使用 zip 和 sum 进行点积计算
                        res_i[j] += sum(a * b for a, b in zip(
                            row_i[k0:k_end], b_col[k0:k_end]))
    
    return result

def benchmark_pure_python(size):
    try:
        # 强制垃圾回收
        gc.collect()
        
        # 记录基准内存使用
        base_memory = get_memory_usage()
        
        # 创建随机矩阵，使用 array 模块
        A = [array('f', (float(np.random.random()) for _ in range(size))) 
             for _ in range(size)]
        B = [array('f', (float(np.random.random()) for _ in range(size))) 
             for _ in range(size)]
        
        # 记录开始状态
        start_time = time.time()
        start_memory = get_memory_usage()
        
        # 执行矩阵乘法
        C = pure_python_matmul_optimized(A, B)
        
        # 确保计算完成
        sum(sum(row) for row in C)  # 强制计算
        
        # 记录结束状态
        end_time = time.time()
        end_memory = get_memory_usage()
        
        # 计算结果
        time_ms = (end_time - start_time) * 1000
        memory_kb = max(end_memory, start_memory) - base_memory
        
        print(f"Pure Python (Optimized) benchmark (size={size}): "
              f"Time={time_ms:.2f}ms, Memory={memory_kb:.2f}KB")
        
        # 保存结果
        results = pd.DataFrame({
            'name': ['matrix_multiplication'],
            'language': ['Pure Python (Opt)'],
            'size': [size],
            'time_ms': [time_ms],
            'memory_kb': [memory_kb]
        })
        
        results_path = '../results/results.csv'
        results.to_csv(results_path, mode='a', header=False, index=False)
        
        # 清理内存
        del A, B, C
        gc.collect()
        
    except Exception as e:
        print(f"Error during Pure Python benchmark (size={size}): {str(e)}")

# ... (NumPy benchmark 函数保持不变) ...


# ... (前面的代码保持不变) ...

def benchmark_numpy(size):
    """NumPy 矩阵乘法基准测试"""
    try:
        # 强制垃圾回收
        gc.collect()
        
        # 记录基准内存使用
        base_memory = get_memory_usage()
        
        # 创建随机矩阵
        A = np.random.rand(size, size).astype(np.float32)
        B = np.random.rand(size, size).astype(np.float32)
        
        # 记录开始状态
        start_time = time.time()
        start_memory = get_memory_usage()
        
        # 执行矩阵乘法
        C = np.matmul(A, B)
        
        # 确保计算完成
        C.sum()  # 强制同步
        
        # 记录结束状态
        end_time = time.time()
        end_memory = get_memory_usage()
        
        # 计算结果
        time_ms = (end_time - start_time) * 1000
        memory_kb = max(end_memory, start_memory) - base_memory
        
        print(f"NumPy benchmark (size={size}): "
              f"Time={time_ms:.2f}ms, Memory={memory_kb:.2f}KB")
        
        # 保存结果
        results = pd.DataFrame({
            'name': ['matrix_multiplication'],
            'language': ['NumPy'],
            'size': [size],
            'time_ms': [time_ms],
            'memory_kb': [memory_kb]
        })
        
        results_path = '../results/results.csv'
        results.to_csv(results_path, mode='a', header=False, index=False)
        
        # 清理内存
        del A, B, C
        gc.collect()
        
    except Exception as e:
        print(f"Error during NumPy benchmark (size={size}): {str(e)}")

if __name__ == '__main__':
    # 纯 Python 测试中等大小的矩阵
    sizes_pure_python = [64, 128, 256, 512]
    
    # NumPy 测试所有大小
    sizes_numpy = [64, 128, 256, 512, 1024]
    
    # 先运行优化后的纯 Python 测试
    print("\nRunning Pure Python (Optimized) benchmarks...")
    for size in sizes_pure_python:
        benchmark_pure_python(size)
    
    # 再运行 NumPy 测试
    print("\nRunning NumPy benchmarks...")
    for size in sizes_numpy:
        benchmark_numpy(size)