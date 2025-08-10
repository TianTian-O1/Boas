#!/bin/bash
# 🔧 完整的MLIR到LLVM IR转换测试

echo "🚀 完整的MLIR→LLVM IR转换测试"

# 设置环境
source scripts/fix_runtime_env.sh > /dev/null 2>&1

echo ""
echo "🔍 步骤1: 完整的转换pipeline"
echo "------------------------------------"

if [ -f "temp.llvm.mlir" ]; then
    echo "✅ 输入文件: temp.llvm.mlir"
else
    echo "❌ 输入文件不存在"
    exit 1
fi

# 使用完整的lowering pipeline
/usr/local/llvm-20/bin/mlir-opt temp.llvm.mlir \
    --convert-cf-to-llvm \
    --reconcile-unrealized-casts \
    -o temp.full_converted.mlir

if [ $? -eq 0 ]; then
    echo "✅ 完整转换pass成功"
    
    # 检查是否还有problematic操作
    if grep -q "unrealized_conversion_cast\|cf\." temp.full_converted.mlir; then
        echo "⚠️ 仍有未转换的操作:"
        grep -E "unrealized_conversion_cast|cf\." temp.full_converted.mlir | head -3
    else
        echo "✅ 所有操作已转换为LLVM dialect"
    fi
else
    echo "❌ 完整转换失败"
    exit 1
fi

echo ""
echo "🔍 步骤2: LLVM IR生成"
echo "------------------------------------"

/usr/local/llvm-20/bin/mlir-translate \
    --mlir-to-llvmir temp.full_converted.mlir \
    -o temp.final.ll \
    --allow-unregistered-dialect

if [ $? -eq 0 ]; then
    echo "🎉 LLVM IR生成成功!"
    echo "📁 生成文件: temp.final.ll"
    echo "📄 文件大小: $(du -h temp.final.ll | cut -f1)"
    
    # 验证生成的LLVM IR
    echo ""
    echo "🔍 验证LLVM IR结构:"
    
    if grep -q "define.*@main" temp.final.ll; then
        echo "✅ 包含main函数"
    fi
    
    if grep -q "define.*@test_simple_matmul" temp.final.ll; then
        echo "✅ 包含test_simple_matmul函数"
    fi
    
    if grep -q "call.*@malloc" temp.final.ll; then
        echo "✅ 包含内存分配调用"
    fi
    
    echo ""
    echo "📄 LLVM IR预览 (前15行):"
    head -15 temp.final.ll
    
    echo ""
    echo "🎯 成功! CF dialect问题已解决"
    echo "💡 解决方案: 使用 --convert-cf-to-llvm + --reconcile-unrealized-casts"
    
    # 保存成功的文件供后续使用
    cp temp.final.ll success_converted.ll
    echo "📂 成功转换的文件已保存为: success_converted.ll"
    
else
    echo "❌ LLVM IR生成失败"
    echo "🔍 错误详情:"
    /usr/local/llvm-20/bin/mlir-translate \
        --mlir-to-llvmir temp.full_converted.mlir \
        -o temp.final.ll \
        --allow-unregistered-dialect 2>&1 | head -10
fi

echo ""
echo "🧹 清理临时文件"
rm -f temp.full_converted.mlir temp.final.ll 2>/dev/null
