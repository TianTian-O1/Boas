#!/bin/bash
# BOAS 快速编译和测试脚本
# Quick build and test script for BOAS

echo "========================================"
echo "BOAS 编译器构建和测试"
echo "========================================"

# 1. 设置环境变量
echo "1. 设置环境变量..."
export LLVM_INSTALL_PATH="/root/llvm-project/llvm_build"
export PATH="$LLVM_INSTALL_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$LLVM_INSTALL_PATH/lib:$LD_LIBRARY_PATH"

# 2. 创建构建目录
echo "2. 创建构建目录..."
if [ ! -d "build" ]; then
    mkdir build
fi

# 3. 配置CMake
echo "3. 配置CMake..."
cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_DIR="$LLVM_INSTALL_PATH/lib/cmake/llvm" \
    -DMLIR_DIR="$LLVM_INSTALL_PATH/lib/cmake/mlir" \
    -DCMAKE_CXX_STANDARD=17

# 4. 编译BOAS
echo "4. 编译BOAS编译器..."
make -j$(nproc)

if [ $? -ne 0 ]; then
    echo "❌ 编译失败！"
    exit 1
fi

echo "✅ 编译成功！"

# 5. 创建测试文件
echo "5. 创建测试文件..."
cd ..

cat > test_matmul.bs << 'EOF'
import tensor

def main():
    print("BOAS 矩阵乘法测试")
    print("=" * 40)
    
    # 测试 2x2 矩阵
    print("\n1. 测试 2x2 矩阵:")
    A = tensor.create(2, 2, [1, 2, 3, 4])
    B = tensor.create(2, 2, [5, 6, 7, 8])
    C = tensor.matmul(A, B)
    print("A =", A)
    print("B =", B)
    print("C = A × B =", C)
    
    # 测试随机矩阵
    print("\n2. 测试 64x64 随机矩阵:")
    A64 = tensor.random(64, 64)
    B64 = tensor.random(64, 64)
    C64 = tensor.matmul(A64, B64)
    print("完成 64x64 矩阵乘法")
    
    print("\n3. 测试 256x256 随机矩阵:")
    A256 = tensor.random(256, 256)
    B256 = tensor.random(256, 256)
    C256 = tensor.matmul(A256, B256)
    print("完成 256x256 矩阵乘法")
    
    print("\n4. 测试 512x512 随机矩阵:")
    A512 = tensor.random(512, 512)
    B512 = tensor.random(512, 512)
    C512 = tensor.matmul(A512, B512)
    print("完成 512x512 矩阵乘法")
    
    print("\n✅ 所有测试完成!")

if __name__ == "__main__":
    main()
EOF

# 6. 编译测试文件
echo "6. 编译测试文件..."
./build/boas-compiler test_matmul.bs -o test_matmul

if [ $? -ne 0 ]; then
    echo "❌ 测试文件编译失败！"
    exit 1
fi

echo "✅ 测试文件编译成功！"

# 7. 运行测试
echo "7. 运行测试..."
./test_matmul

echo ""
echo "========================================"
echo "完成！BOAS 编译器已准备就绪"
echo "========================================"
echo ""
echo "你可以使用以下命令编译和运行 BOAS 程序："
echo "  ./build/boas-compiler <源文件.bs> -o <输出文件>"
echo "  ./<输出文件>"
echo ""
echo "示例："
echo "  ./build/boas-compiler examples/hello_world.bs -o hello"
echo "  ./hello"