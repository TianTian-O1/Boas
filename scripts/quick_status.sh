#!/bin/bash
# 快速状态检查脚本

echo "⚡ LLVM快速状态:"
echo "构建大小: $(du -sh /tmp/llvm-20/build | cut -f1)"
echo "编译进程: $(ps aux | grep 'c++.*llvm' | grep -v grep | wc -l)个"
echo "已完成: MLIR工具$(find /tmp/llvm-20/build/bin -name '*mlir*' -executable | wc -l)个, Clang工具$(find /tmp/llvm-20/build/bin -name '*clang*' | wc -l)个"

# 检查关键文件
if [ -f "/tmp/llvm-20/build/bin/clang-20" ]; then
    echo "✅ clang-20: $(du -sh /tmp/llvm-20/build/bin/clang-20 | cut -f1)"
fi

if [ -f "/tmp/llvm-20/build/bin/mlir-opt" ]; then
    echo "✅ mlir-opt: $(du -sh /tmp/llvm-20/build/bin/mlir-opt | cut -f1)"
else
    echo "⏳ mlir-opt: 编译中"
fi

echo "时间: $(date +%H:%M:%S)"

# 快速状态检查脚本

echo "⚡ LLVM快速状态:"
echo "构建大小: $(du -sh /tmp/llvm-20/build | cut -f1)"
echo "编译进程: $(ps aux | grep 'c++.*llvm' | grep -v grep | wc -l)个"
echo "已完成: MLIR工具$(find /tmp/llvm-20/build/bin -name '*mlir*' -executable | wc -l)个, Clang工具$(find /tmp/llvm-20/build/bin -name '*clang*' | wc -l)个"

# 检查关键文件
if [ -f "/tmp/llvm-20/build/bin/clang-20" ]; then
    echo "✅ clang-20: $(du -sh /tmp/llvm-20/build/bin/clang-20 | cut -f1)"
fi

if [ -f "/tmp/llvm-20/build/bin/mlir-opt" ]; then
    echo "✅ mlir-opt: $(du -sh /tmp/llvm-20/build/bin/mlir-opt | cut -f1)"
else
    echo "⏳ mlir-opt: 编译中"
fi

echo "时间: $(date +%H:%M:%S)"
