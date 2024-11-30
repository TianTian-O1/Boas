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
if [ -f "../results/results.csv" ]; then
    rm -f "../results/results.csv"
fi

# 创建结果文件头
echo "name,language,size,time_ms" > "../results/results.csv"

# 编译并运行C++基准测试
echo "Building and running C++ benchmarks..."
mkdir -p "../build"
cd "../build"

# Windows和Unix的cmake命令不同
if [ "$OSTYPE" = "msys" ] || [ "$OSTYPE" = "win32" ] || [ "$OSTYPE" = "cygwin" ]; then
    cmake -G "Visual Studio 17 2022" ..
else
    cmake ..
fi

# Windows和Unix的make命令不同
if [ "$OSTYPE" = "msys" ] || [ "$OSTYPE" = "win32" ] || [ "$OSTYPE" = "cygwin" ]; then
    cmake --build . --config Release
else
    make
fi

# 运行编译后的程序
if [ "$OSTYPE" = "msys" ] || [ "$OSTYPE" = "win32" ] || [ "$OSTYPE" = "cygwin" ]; then
    ./Release/cpp_bench.exe
else
    ./cpp_bench
fi

# 运行Python基准测试
echo "Running Python benchmarks..."
cd "../src"
if [ -f "python_bench.py" ]; then
    python3 python_bench.py
else
    echo "python_bench.py not found"
fi

# 运行Boas基准测试
echo "Running Boas benchmarks..."
cd "../../"  # 返回到项目根目录
if [ -f "benchmark/src/boas_bench.bs" ]; then
    if [ "$OSTYPE" = "msys" ] || [ "$OSTYPE" = "win32" ] || [ "$OSTYPE" = "cygwin" ]; then
        build/Release/matrix-compiler.exe --run benchmark/src/boas_bench.bs
    else
        build/matrix-compiler --run benchmark/src/boas_bench.bs
    fi
fi

# 生成图表
echo "Generating plots..."
cd "benchmark/scripts"
python3 plot.py

echo "Benchmark complete! Results are in ../results/comparison.png"