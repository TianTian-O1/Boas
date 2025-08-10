#!/bin/bash
# LLVM 20 编译完成监控和自动安装脚本

echo "=== LLVM 20 编译监控启动 ==="
echo "开始时间: $(date)"
echo

BUILD_DIR="/tmp/llvm-20/build"
INSTALL_DIR="/usr/local/llvm-20"

# 监控编译进度
monitor_compilation() {
    while true; do
        # 检查make进程是否还在运行
        MAKE_PROCESSES=$(ps aux | grep "make.*llvm" | grep -v grep | wc -l)
        COMPILE_PROCESSES=$(ps aux | grep "c++.*llvm" | grep -v grep | wc -l)
        
        if [ $MAKE_PROCESSES -eq 0 ] && [ $COMPILE_PROCESSES -eq 0 ]; then
            echo "✅ LLVM 20编译完成！"
            break
        fi
        
        # 显示进度
        BUILD_SIZE=$(du -sh $BUILD_DIR | cut -f1)
        echo "⏳ $(date +%H:%M:%S) - 编译中... 大小: $BUILD_SIZE, Make进程: $MAKE_PROCESSES, 编译进程: $COMPILE_PROCESSES"
        
        sleep 30
    done
}

# 安装LLVM 20
install_llvm() {
    echo "🚀 开始安装LLVM 20到 $INSTALL_DIR"
    
    cd $BUILD_DIR
    if make install; then
        echo "✅ LLVM 20安装成功！"
        
        # 验证安装
        echo "📋 验证安装的工具:"
        for tool in clang mlir-opt mlir-translate llc; do
            if [ -f "$INSTALL_DIR/bin/$tool" ]; then
                echo "✓ $tool - $($INSTALL_DIR/bin/$tool --version | head -1)"
            else
                echo "✗ $tool - 未找到"
            fi
        done
        
        return 0
    else
        echo "❌ LLVM 20安装失败"
        return 1
    fi
}

# 测试Boas编译
test_boas_compilation() {
    echo "🧪 测试Boas项目编译"
    
    cd /root/Boas/Boas-linux
    rm -rf build
    mkdir build
    cd build
    
    if cmake .. -DLLVM_INSTALL_PREFIX=$INSTALL_DIR -DCMAKE_BUILD_TYPE=Release; then
        echo "✅ CMake配置成功"
        
        if make -j4; then
            echo "🎉 Boas项目编译成功！"
            
            # 测试NPU功能
            echo "🎯 测试NPU矩阵乘法..."
            if [ -f "./test-full-pipeline" ]; then
                echo "import tensor
def test():
    var A = tensor.random(128, 128)
    var B = tensor.random(128, 128) 
    var C = tensor.matmul(A, B)
    return C
def main():
    var result = test()
    return 0" > /tmp/boas_test.bs
                
                if timeout 30 ./test-full-pipeline /tmp/boas_test.bs; then
                    echo "🎉 Boas NPU测试成功！"
                else
                    echo "⚠️  Boas NPU测试未完全成功（可能需要进一步调试）"
                fi
            fi
            
            return 0
        else
            echo "❌ Boas项目编译失败"
            return 1
        fi
    else
        echo "❌ CMake配置失败"
        return 1
    fi
}

# 主流程
main() {
    echo "开始监控LLVM 20编译进度..."
    monitor_compilation
    
    echo "开始安装LLVM 20..."
    if install_llvm; then
        echo "开始测试Boas编译..."
        test_boas_compilation
    fi
    
    echo "=== 完成时间: $(date) ==="
}

# 运行主流程
main 2>&1 | tee /root/Boas/Boas-linux/llvm20_install.log
