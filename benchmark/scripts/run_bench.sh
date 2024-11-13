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
if [ -f "../../boas" ]; then
    ../../boas boas_bench.bs
else
    echo "Warning: Boas compiler not found, skipping Boas benchmarks"
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
