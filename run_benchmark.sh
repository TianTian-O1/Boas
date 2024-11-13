#!/bin/bash

# 创建基准测试目录结构
mkdir -p benchmark/{data,results,scripts,src,include}

# 创建 benchmark.h
cat > benchmark/include/benchmark.h << 'EOL'
#pragma once
#include <chrono>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>

class Benchmark {
public:
    struct Result {
        std::string name;
        std::string language;
        size_t size;
        double time_ms;
        size_t memory_kb;
    };

    static void run(const std::string& name, 
                   const std::string& language,
                   size_t size,
                   std::function<void()> func) {
        // 记录开始时间
        auto start = std::chrono::high_resolution_clock::now();
        
        // 记录开始内存
        size_t start_memory = getCurrentMemory();
        
        // 运行测试
        func();
        
        // 记录结束时间和内存
        auto end = std::chrono::high_resolution_clock::now();
        size_t end_memory = getCurrentMemory();
        
        // 计算结果
        double time_ms = std::chrono::duration_cast<std::chrono::microseconds>
            (end - start).count() / 1000.0;
        size_t memory_kb = (end_memory - start_memory) / 1024;
        
        // 保存结果
        Result result{name, language, size, time_ms, memory_kb};
        saveResult(result);
    }

private:
    static size_t getCurrentMemory() {
        #ifdef __APPLE__
            // MacOS implementation
            struct rusage rusage;
            getrusage(RUSAGE_SELF, &rusage);
            return (size_t)(rusage.ru_maxrss);
        #elif __linux__
            // Linux implementation
            FILE* file = fopen("/proc/self/status", "r");
            if (file) {
                char line[128];
                while (fgets(line, 128, file) != NULL) {
                    if (strncmp(line, "VmRSS:", 6) == 0) {
                        size_t size;
                        sscanf(line, "VmRSS: %lu", &size);
                        fclose(file);
                        return size * 1024;
                    }
                }
                fclose(file);
            }
            return 0;
        #else
            return 0;
        #endif
    }

    static void saveResult(const Result& result) {
        std::ofstream file("../results/results.csv", std::ios::app);
        if (file.is_open()) {
            file << result.name << ","
                 << result.language << ","
                 << result.size << ","
                 << result.time_ms << ","
                 << result.memory_kb << "\n";
        }
    }
};
EOL

# 创建 cpp_bench.cpp
cat > benchmark/src/cpp_bench.cpp << 'EOL'
#include "../include/benchmark.h"
#include <vector>
#include <random>
#include <omp.h>

void matrix_multiply(const std::vector<std::vector<float>>& A,
                    const std::vector<std::vector<float>>& B,
                    std::vector<std::vector<float>>& C) {
    int m = A.size();
    int k = A[0].size();
    int n = B[0].size();

    // 优化版本的矩阵乘法实现
    const int BLOCK_SIZE = 32;

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < m; i += BLOCK_SIZE) {
        for (int j = 0; j < n; j += BLOCK_SIZE) {
            for (int p = 0; p < k; p += BLOCK_SIZE) {
                for (int ii = i; ii < std::min(i + BLOCK_SIZE, m); ii++) {
                    for (int jj = j; jj < std::min(j + BLOCK_SIZE, n); jj++) {
                        float sum = 0.0f;
                        for (int pp = p; pp < std::min(p + BLOCK_SIZE, k); pp++) {
                            sum += A[ii][pp] * B[pp][jj];
                        }
                        C[ii][jj] += sum;
                    }
                }
            }
        }
    }
}

int main() {
    std::vector<int> sizes = {64, 128, 256, 512, 1024};
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    for (int size : sizes) {
        // 创建测试矩阵
        std::vector<std::vector<float>> A(size, std::vector<float>(size));
        std::vector<std::vector<float>> B(size, std::vector<float>(size));
        std::vector<std::vector<float>> C(size, std::vector<float>(size, 0.0f));

        // 初始化矩阵
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                A[i][j] = dis(gen);
                B[i][j] = dis(gen);
            }
        }

        // 运行基准测试
        Benchmark::run("matrix_multiplication", "C++", size, 
            [&]() { matrix_multiply(A, B, C); });
    }

    return 0;
}
EOL

# 创建 python_bench.py
cat > benchmark/src/python_bench.py << 'EOL'
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
    results = pd.DataFrame({
        'name': ['matrix_multiplication'],
        'language': ['Python'],
        'size': [size],
        'time_ms': [time_ms],
        'memory_kb': [memory_kb]
    })
    results.to_csv('../results/results.csv', mode='a', header=False, index=False)

if __name__ == '__main__':
    sizes = [64, 128, 256, 512, 1024]
    for size in sizes:
        benchmark_numpy(size)
EOL

# 创建 boas_bench.bs
cat > benchmark/src/boas_bench.bs << 'EOL'
import tensor
from tensor import matmul

def benchmark(size: int):
    # 创建随机矩阵
    A = tensor.random(size, size)
    B = tensor.random(size, size)
    
    # 执行矩阵乘法
    C = matmul(A, B)

def main():
    sizes = [64, 128, 256, 512, 1024]
    for size in sizes:
        benchmark(size)
EOL

# 创建绘图脚本
cat > benchmark/scripts/plot.py << 'EOL'
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_results():
    # 读取结果
    df = pd.read_csv('../results/results.csv')
    
    # 创建性能对比图
    plt.figure(figsize=(12, 6))
    
    # 时间对比
    plt.subplot(1, 2, 1)
    sns.lineplot(data=df, x='size', y='time_ms', hue='language', marker='o')
    plt.title('Execution Time Comparison')
    plt.xlabel('Matrix Size')
    plt.ylabel('Time (ms)')
    plt.yscale('log')
    
    # 内存对比
    plt.subplot(1, 2, 2)
    sns.lineplot(data=df, x='size', y='memory_kb', hue='language', marker='o')
    plt.title('Memory Usage Comparison')
    plt.xlabel('Matrix Size')
    plt.ylabel('Memory (KB)')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('../results/comparison.png')
    plt.close()

if __name__ == '__main__':
    plot_results()
EOL

# 创建运行脚本
cat > benchmark/scripts/run_bench.sh << 'EOL'
#!/bin/bash

# 清理旧结果
rm -f ../results/results.csv

# 创建结果文件头
echo "name,language,size,time_ms,memory_kb" > ../results/results.csv

# 编译并运行C++基准测试
echo "Building and running C++ benchmarks..."
mkdir -p ../build
cd ../build
cmake ..
make
./cpp_bench

# 运行Python基准测试
echo "Running Python benchmarks..."
cd ../src
python3 python_bench.py

# 运行Boas基准测试
echo "Running Boas benchmarks..."
../../boas boas_bench.bs

# 生成图表
echo "Generating plots..."
cd ../scripts
python3 plot.py

echo "Benchmark complete! Results are in ../results/comparison.png"
EOL

# 创建CMakeLists.txt
cat > benchmark/CMakeLists.txt << 'EOL'
cmake_minimum_required(VERSION 3.10)
project(matrix_benchmark)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_executable(cpp_bench src/cpp_bench.cpp)

if(OpenMP_CXX_FOUND)
    target_link_libraries(cpp_bench PUBLIC OpenMP::OpenMP_CXX)
endif()
EOL

# 设置执行权限
chmod +x benchmark/scripts/run_bench.sh

echo "Benchmark environment setup complete!"
echo "To run benchmarks, execute: cd benchmark/scripts && ./run_bench.sh"

# 安装Python依赖
pip install numpy pandas matplotlib seaborn psutil
