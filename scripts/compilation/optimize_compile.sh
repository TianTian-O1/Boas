#!/bin/bash
# 🚀 优化版MLIR转换pipeline

echo "🔧 应用高级MLIR优化..."

/usr/local/llvm-20/bin/mlir-opt temp.llvm.mlir \
    --convert-linalg-to-loops \
    --loop-invariant-code-motion \
    --affine-loop-unroll="unroll-factor=8" \
    --affine-loop-fusion \
    --canonicalize \
    --cse \
    --convert-scf-to-cf \
    --convert-cf-to-llvm \
    --reconcile-unrealized-casts \
    -o temp.optimized.mlir

if [ $? -eq 0 ]; then
    echo "✅ 优化pipeline成功"
    
    echo "🔧 生成优化的LLVM IR..."
    /usr/local/llvm-20/bin/mlir-translate \
        --mlir-to-llvmir temp.optimized.mlir \
        -o temp.optimized.ll \
        --allow-unregistered-dialect
        
    if [ $? -eq 0 ]; then
        echo "✅ 优化LLVM IR生成成功"
        
        echo "🔧 生成优化汇编(-O3 + 向量化)..."
        /usr/local/llvm-20/bin/llc temp.optimized.ll \
            -o temp.optimized.s \
            -O3 \
            -enable-unsafe-fp-math \
            
            
        if [ $? -eq 0 ]; then
            echo "✅ 优化汇编生成成功"
            
            echo "🔧 链接优化版本..."
            gcc temp.optimized.s -o boas_npu_optimized \
                -O3  -march=native \
                -L/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64 \
                -lm
                
            if [ $? -eq 0 ]; then
                echo "🎉 优化版本编译成功: boas_npu_optimized"
                ls -la boas_npu_optimized
            else
                echo "❌ 链接失败"
            fi
        else
            echo "❌ 汇编生成失败"
        fi
    else
        echo "❌ LLVM IR生成失败"
    fi
else
    echo "❌ 优化pipeline失败"
fi
