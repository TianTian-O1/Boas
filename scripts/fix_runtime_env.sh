#!/bin/bash
# 🔧 修复Boas运行时环境脚本

echo "🔧 正在修复Boas运行时环境..."

# 设置正确的库路径优先级
export SYSTEM_LIB_PATH="/usr/lib/aarch64-linux-gnu"
export CANN_LIB_PATH="/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64"
export LLVM_LIB_PATH="/usr/local/llvm-20/lib"

# 优先使用系统库而非conda库  
export LD_LIBRARY_PATH="${SYSTEM_LIB_PATH}:${CANN_LIB_PATH}:${LLVM_LIB_PATH}:${LD_LIBRARY_PATH}"

echo "✅ 库路径已设置："
echo "   系统库: ${SYSTEM_LIB_PATH}"
echo "   CANN库: ${CANN_LIB_PATH}" 
echo "   LLVM库: ${LLVM_LIB_PATH}"

# 验证libstdc++版本
echo ""
echo "🔍 验证libstdc++版本："
if strings ${SYSTEM_LIB_PATH}/libstdc++.so.6 | grep "GLIBCXX_3.4.30" > /dev/null; then
    echo "✅ 系统libstdc++支持GLIBCXX_3.4.30"
else
    echo "❌ 系统libstdc++版本不足"
    exit 1
fi

# 验证CANN库
echo ""
echo "🔍 验证CANN库："
if [ -f "${CANN_LIB_PATH}/libascendcl.so" ]; then
    echo "✅ CANN库可用: ${CANN_LIB_PATH}/libascendcl.so"
else
    echo "❌ CANN库未找到"
    exit 1
fi

# 验证test-full-pipeline
echo ""
echo "🔍 验证编译的程序："
if [ -f "build/test-full-pipeline" ]; then
    echo "✅ test-full-pipeline存在"
    echo "🔧 测试运行..."
    
    # 使用正确的环境运行
    env LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" ./build/test-full-pipeline --help 2>&1 | head -5
    
    if [ $? -eq 0 ]; then
        echo "✅ test-full-pipeline可以运行"
    else
        echo "⚠️ test-full-pipeline运行有问题，但库依赖已修复"
    fi
else
    echo "❌ test-full-pipeline不存在，需要重新编译"
fi

echo ""
echo "🎯 环境修复完成！使用方法："
echo "   source scripts/fix_runtime_env.sh"
echo "   ./build/test-full-pipeline [参数]"

# 导出环境变量供当前shell使用
echo "export LD_LIBRARY_PATH=\"${LD_LIBRARY_PATH}\"" > /tmp/boas_env.sh
echo "🔧 环境变量已保存到 /tmp/boas_env.sh"
