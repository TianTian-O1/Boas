#!/bin/bash
# 🚀 直接测试NPU功能脚本 - 绕过mlir-translate问题

echo "🚀 测试Boas NPU功能 (绕过LLVM IR生成)"
echo "=========================================="

# 设置环境
source scripts/fix_runtime_env.sh > /dev/null 2>&1

echo ""
echo "🔍 测试1: MLIR代码生成和NPU属性"
echo "------------------------------------"

# 生成MLIR并检查NPU属性
echo "生成MLIR代码..."
./build/test-full-pipeline --build test_fix_compilation.bs > test_output.log 2>&1

if grep -q "npu_optimized" test_output.log; then
    echo "✅ NPU优化属性生成成功"
    grep "boas\.backend.*npu_optimized" test_output.log | head -1
else
    echo "❌ NPU优化属性未找到"
fi

if grep -q "Successfully initialized with.*device" test_output.log; then
    echo "✅ CANN运行时初始化成功"
    grep "Successfully initialized" test_output.log | head -1
else
    echo "❌ CANN运行时初始化失败"
fi

if grep -q "linalg.matmul" test_output.log; then
    echo "✅ 矩阵乘法MLIR代码生成"
    grep -A 1 -B 1 "linalg.matmul" test_output.log | head -3
else
    echo "❌ 矩阵乘法MLIR代码未生成"
fi

echo ""
echo "🔍 测试2: NPU设备状态检查"
echo "------------------------------------"

# 检查NPU设备状态
if command -v npu-smi &> /dev/null; then
    echo "🔧 检查NPU设备状态:"
    npu-smi info | head -10
else
    echo "⚠️ npu-smi工具未找到，无法检查NPU状态"
fi

echo ""
echo "🔍 测试3: CANN库链接验证"
echo "------------------------------------"

if ldd build/test-full-pipeline | grep -q "libascendcl"; then
    echo "✅ CANN库链接成功"
    ldd build/test-full-pipeline | grep ascendcl
else
    echo "❌ CANN库链接失败"
fi

echo ""
echo "📊 测试结果汇总"
echo "=========================================="

# 统计成功项
SUCCESS_COUNT=0
TOTAL_TESTS=6

echo "检查项目 | 状态"
echo "--------|------"

# 检查各项功能
if grep -q "npu_optimized" test_output.log; then
    echo "NPU属性生成 | ✅ 成功"
    ((SUCCESS_COUNT++))
else
    echo "NPU属性生成 | ❌ 失败"
fi

if grep -q "Successfully initialized" test_output.log; then
    echo "CANN初始化 | ✅ 成功"
    ((SUCCESS_COUNT++))
else
    echo "CANN初始化 | ❌ 失败"
fi

if grep -q "linalg.matmul" test_output.log; then
    echo "MLIR生成 | ✅ 成功"
    ((SUCCESS_COUNT++))
else
    echo "MLIR生成 | ❌ 失败"
fi

if ldd build/test-full-pipeline | grep -q "libascendcl"; then
    echo "CANN库链接 | ✅ 成功"
    ((SUCCESS_COUNT++))
else
    echo "CANN库链接 | ❌ 失败"
fi

if [ -f "build/test-full-pipeline" ]; then
    echo "编译成功 | ✅ 成功"
    ((SUCCESS_COUNT++))
else
    echo "编译成功 | ❌ 失败"
fi

if [ ! -z "$LD_LIBRARY_PATH" ] && echo "$LD_LIBRARY_PATH" | grep -q "aarch64-linux-gnu"; then
    echo "环境配置 | ✅ 成功"
    ((SUCCESS_COUNT++))
else
    echo "环境配置 | ❌ 失败"
fi

echo ""
echo "🎯 总体结果: $SUCCESS_COUNT/$TOTAL_TESTS 项成功"

if [ $SUCCESS_COUNT -ge 4 ]; then
    echo "🎉 核心NPU功能基本就绪！"
    echo "📋 下一步: 实现端到端NPU kernel调用"
else
    echo "⚠️ 需要解决更多基础问题"
fi

echo ""
echo "🔧 待解决问题:"
if ! grep -q "npu_optimized" test_output.log; then
    echo "- NPU代码生成优化"
fi
echo "- mlir-translate的cf dialect注册"
echo "- 端到端NPU kernel执行"
echo "- 性能测试和对比"

echo ""
echo "📁 日志文件: test_output.log"

rm -f test_output.log
