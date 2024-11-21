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
        output_file=$(mktemp)
        memory_file=$(mktemp)
        base_memory_file=$(mktemp)
        
        echo "Measuring base memory usage..."
        # 测量基准内存使用
        sleep 1
        ps -o rss= -p $$ > "$base_memory_file"
        base_memory=$(cat "$base_memory_file")
        
        echo "Executing program..."
        
        # 运行程序并捕获输出和进程ID
        {
            ../../build/matrix-compiler --run boas_bench.bs 2>&1 &
            compiler_pid=$!
            
            # 监控进程内存使用 (提高采样频率到0.1ms)
            while kill -0 $compiler_pid 2>/dev/null; do
                ps -o rss= -p $compiler_pid >> "$memory_file"
                sleep 0.0001  # 每0.1ms采样一次
            done
        } | grep -E "^(202[0-9]|[0-9]$)" > "$output_file"
        
        # 使用Python处理输出和内存数据
        python3 -c '
import sys
from datetime import datetime
import re
import numpy as np

def parse_timestamp(ts_str):
    return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f").timestamp()

# 读取基准内存
with open(sys.argv[3], "r") as f:
    base_memory = int(f.readline().strip())

# 读取内存记录并减去基准内存
with open(sys.argv[2], "r") as f:
    memory_samples = [max(0, int(line.strip()) - base_memory) for line in f if line.strip()]

# 如果样本太少，插值填充
if len(memory_samples) < 1000:
    memory_samples = np.interp(
        np.linspace(0, len(memory_samples)-1, 1000),
        np.arange(len(memory_samples)),
        memory_samples
    ).tolist()

timestamps = []
numbers = []

# 读取时间戳和数字
with open(sys.argv[1], "r") as f:
    for line in f:
        line = line.strip()
        if re.match(r"\d{4}-\d{2}-\d{2}", line):  # 时间戳
            timestamps.append(parse_timestamp(line))
        elif line.isdigit():  # 数字
            numbers.append(int(line))

# 处理每对时间戳之间的差值，使用滑动窗口
matrix_sizes = [64, 128, 256, 512, 1024]
window_size = int(len(memory_samples) * 0.1)  # 使用10%数据作为窗口

for i in range(5):  # 5个矩阵大小
    size = matrix_sizes[i]
    idx = i * 2  # 每个测试有两个数字
    if idx + 1 < len(timestamps):
        start_time = timestamps[idx]
        end_time = timestamps[idx + 1]
        time_diff = (end_time - start_time) * 1000  # 转换为毫秒
        
        # 使用滑动窗口找出最大内存使用
        max_memory = 0
        if len(memory_samples) >= window_size:
            for j in range(len(memory_samples) - window_size + 1):
                window_avg = np.mean(memory_samples[j:j+window_size])
                max_memory = max(max_memory, window_avg)
        else:
            max_memory = max(memory_samples) if memory_samples else 0
            
        # 输出结果
        print(f"{size},{time_diff},{max_memory:.1f}")

' "$output_file" "$memory_file" "$base_memory_file" > "$output_file.times"
        
        # 读取并显示结果
        while IFS=, read -r size time_ms memory_kb; do
            echo "matrix_multiplication,Boas,$size,$time_ms,$memory_kb" >> ../results/results.csv
            echo "Matrix size $size completed in $time_ms ms, memory: $memory_kb KB"
        done < "$output_file.times"
        
        # 清理临时文件
        rm "$output_file"
        rm "$output_file.times"
        rm "$memory_file"
        rm "$base_memory_file"
        
        echo "Benchmark completed successfully"
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