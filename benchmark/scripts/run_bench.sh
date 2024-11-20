#!/bin/bash

# 检查是否在正确的目录
if [ ! -d "../results" ]; then
    echo "Error: Please run this script from the benchmark/scripts directory"
    exit 1
fi

# 清理旧结果
rm -f ../results/results.csv

# 创建结果文件头
echo "name,language,size,time_ms,memory_kb" > ../results/results.csv

# 编译并运行C++基准测试
echo "Building and running C++ benchmarks..."
mkdir -p ../build
cd ../build
if ! cmake ..; then
    echo "CMake configuration failed"
    exit 1
fi

if ! make; then
    echo "Make failed"
    exit 1
fi

if [ -f "./cpp_bench" ]; then
    ./cpp_bench
else
    echo "cpp_bench executable not found"
    exit 1
fi

# 运行Python基准测试
echo "Running Python benchmarks..."
cd ../src
if [ -f "python_bench.py" ]; then
    python3 python_bench.py
else
    echo "python_bench.py not found"
fi

# 运行Boas基准测试
echo "Running Boas benchmarks..."
cd ../src
if [ -f "boas_bench.bs" ]; then
    if [ -f "../../build/matrix-compiler" ]; then
        for size in 64 128 256 512 1024; do
            # 记录开始时间和内存
            start_time=$(date +%s.%N)
            start_memory=$(ps -o rss= -p $$)
            
            # 运行测试
            ../../build/matrix-compiler --run boas_bench.bs
            
            # 记录结束时间和内存
            end_time=$(date +%s.%N)
            end_memory=$(ps -o rss= -p $$)
            
            # 计算时间和内存使用
            time_ms=$(echo "($end_time - $start_time) * 1000" | bc)
            memory_kb=$(echo "$end_memory - $start_memory" | bc)
            
            # 添加到结果文件
            echo "matrix_multiplication,Boas,$size,$time_ms,$memory_kb" >> ../results/results.csv
        done
    else
        echo "Warning: matrix-compiler not found, skipping Boas benchmarks"
    fi
else
    echo "Warning: boas_bench.bs not found"
fi

# 生成图表
echo "Generating plots..."
cd ../scripts
if [ -f "plot.py" ]; then
    python3 plot.py
    echo "Benchmark complete! Results are in ../results/comparison.png"
else
    echo "Error: plot.py not found"
    exit 1
fi
