#!/bin/bash
# 🔧 测试CF dialect转换的简单脚本

echo "🧪 测试CF dialect到LLVM的转换方案"

# 设置环境
source ../scripts/fix_runtime_env.sh > /dev/null 2>&1

echo ""
echo "🔍 步骤1: 检查生成的MLIR文件中的cf dialect"
echo "------------------------------------"

if [ -f "temp.llvm.mlir" ]; then
    echo "✅ 找到MLIR文件: temp.llvm.mlir"
    
    # 检查是否包含cf dialect
    if grep -q "cf\." temp.llvm.mlir; then
        echo "✅ 文件包含cf dialect操作:"
        grep "cf\." temp.llvm.mlir | head -3
    else
        echo "❌ 文件不包含cf dialect操作"
        exit 1
    fi
else
    echo "❌ MLIR文件不存在，请先运行Boas编译"
    exit 1
fi

echo ""
echo "🔍 步骤2: 使用mlir-opt转换cf dialect"
echo "------------------------------------"

# 尝试使用mlir-opt转换cf dialect到LLVM
/usr/local/llvm-20/bin/mlir-opt temp.llvm.mlir \
    --convert-cf-to-llvm \
    -o temp.cf_converted.mlir

if [ $? -eq 0 ]; then
    echo "✅ CF到LLVM转换成功"
    
    # 检查转换后的文件
    if grep -q "cf\." temp.cf_converted.mlir; then
        echo "⚠️ 转换后仍包含cf dialect"
        grep "cf\." temp.cf_converted.mlir | head -3
    else
        echo "✅ CF dialect已成功转换为LLVM dialect"
    fi
else
    echo "❌ CF到LLVM转换失败"
    exit 1
fi

echo ""
echo "🔍 步骤3: 使用mlir-translate转换LLVM IR"
echo "------------------------------------"

# 尝试使用转换后的文件进行LLVM IR生成
/usr/local/llvm-20/bin/mlir-translate \
    --mlir-to-llvmir temp.cf_converted.mlir \
    -o temp.converted.ll \
    --allow-unregistered-dialect

if [ $? -eq 0 ]; then
    echo "✅ LLVM IR生成成功!"
    echo "📁 生成文件: temp.converted.ll"
    echo "📄 文件大小: $(du -h temp.converted.ll | cut -f1)"
    
    # 显示LLVM IR的前几行
    echo ""
    echo "🔍 生成的LLVM IR预览:"
    head -10 temp.converted.ll
    
    echo ""
    echo "🎉 CF dialect转换方案验证成功!"
    echo "💡 解决方案: 在test_full_pipeline.cpp中添加cf转换步骤"
    
else
    echo "❌ LLVM IR生成失败"
    
    # 检查错误信息
    echo "🔍 错误分析: 检查是否还有未注册的dialect"
    /usr/local/llvm-20/bin/mlir-translate \
        --mlir-to-llvmir temp.cf_converted.mlir \
        -o temp.converted.ll \
        --allow-unregistered-dialect 2>&1 | head -5
fi

echo ""
echo "🧹 清理临时文件"
rm -f temp.cf_converted.mlir temp.converted.ll 2>/dev/null

echo "✅ 测试完成"
