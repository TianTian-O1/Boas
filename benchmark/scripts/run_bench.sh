#!/bin/bash

# 检查必要的命令是否存在
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "Warning: $1 is not installed, skipping $2 benchmarks"
        return 1
    fi
    return 0
}

# 检查是否在正确的目录
if [ ! -d "../results" ]; then
    echo "Error: Please run this script from the benchmark/scripts directory"
    exit 1
fi

# 清理旧结果
rm -f ../results/results.csv

# 创建结果文件头
echo "name,language,size,time_ms" > ../results/results.csv

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

# 运行 Go 基准测试
if check_command "go" "Go"; then
    echo "Running Go benchmarks..."
    cd ../src
    if [ -f "go_bench.go" ]; then
        go run go_bench.go
    else
        echo "Warning: go_bench.go not found"
    fi
fi

# 运行 Java 基准测试
if check_command "javac" "Java"; then
    echo "Running Java benchmarks..."
    cd ../src
    if [ -f "JavaBench.java" ]; then
        javac JavaBench.java
        java JavaBench
        rm -f JavaBench.class
    else
        echo "Warning: JavaBench.java not found"
    fi
fi

# 运行Boas基准测试
echo "Running Boas benchmarks..."
cd ../../  # 返回到项目根目录
if [ -f "benchmark/src/boas_bench.bs" ]; then
    if [ -f "build/matrix-compiler" ]; then
        echo "Executing program..."
        
        # 定义矩阵大小数组
        sizes=(64 128 256 512 1024)
        count=0
        
        # 运行程序并直接处理输出
        build/matrix-compiler --run benchmark/src/boas_bench.bs 2>&1 | while read -r line; do
            if [[ $line =~ "Time taken: "([0-9.]+)" ms" ]]; then
                time_ms=${BASH_REMATCH[1]}
                size=${sizes[$count]}
                echo "matrix_multiplication,Boas,$size,$time_ms" >> benchmark/results/results.csv
                echo "Matrix size $size completed in $time_ms ms"
                ((count++))
            fi
        done

        # 清理临时文件
        rm -f benchmark/src/temp.mlir
        rm -f benchmark/src/temp.llvm.mlir
        rm -f benchmark/src/temp.ll
        rm -f benchmark/src/temp.s
        rm -f benchmark/src/a.out
        
        echo "Benchmark completed successfully"
    else
        echo "Warning: matrix-compiler not found, skipping Boas benchmarks"
    fi
else
    echo "Warning: boas_bench.bs not found"
fi



# 返回到scripts目录以继续执行其他测试
cd benchmark/scripts

# 生成图表
echo "Generating plots..."
if [ -f "plot.py" ]; then
    python3 plot.py
    echo "Benchmark complete! Results are in ../results/comparison.png"
else
    echo "Error: plot.py not found"
    exit 1
fi



# 运行性能分析
echo "Running performance analysis..."
if [ "$(uname)" == "Darwin" ]; then
    echo "Using macOS performance tools..."
    mkdir -p ../results/instruments
    
    cd ../../build  # 切换到包含 matrix-compiler 的目录
    
    # 使用 time 命令进行完整程序分析
    echo "Running full program analysis..."
    /usr/bin/time -l ./matrix-compiler --run ../benchmark/src/boas_bench.bs \
        2> ../benchmark/results/instruments/full_profile.txt
    
    # CPU 采样分析 - 使用后台运行和进程 ID
    echo "Running CPU sampling..."
    ./matrix-compiler --run ../benchmark/src/boas_bench.bs & 
    COMPILER_PID=$!
    sleep 1  # 给进程启动的时间
    
    if ps -p $COMPILER_PID > /dev/null; then
        sample $COMPILER_PID 5 -file ../benchmark/results/instruments/cpu_profile.txt
        wait $COMPILER_PID
    else
        echo "Warning: Could not sample process - it may have finished too quickly"
    fi
    
    cd ../benchmark/scripts  # 切回原目录
    
    # 生成简单的性能报告
    echo "Generating performance report..."
    {
        echo "Performance Analysis Report"
        echo "=========================="
        echo
        echo "Memory Usage:"
        grep "maximum resident set size" ../results/instruments/full_profile.txt
        echo
        echo "CPU Profile Summary:"
        if [ -f "../results/instruments/cpu_profile.txt" ]; then
            head -n 20 ../results/instruments/cpu_profile.txt
        else
            echo "CPU profile not available"
        fi
    } > ../results/instruments/analysis_report.txt
    
    echo "Performance analysis completed. Results in ../results/instruments/"
else
    echo "Warning: No performance analysis tools found"
fi