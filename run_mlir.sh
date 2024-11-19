#!/bin/bash

# 检查输入文件
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <input>"
    exit 1
fi

# 设置 LLVM 工具路径
LLVM_PATH="/Users/mac/llvm-install"
MLIR_OPT="${LLVM_PATH}/bin/mlir-opt"
MLIR_TRANSLATE="${LLVM_PATH}/bin/mlir-translate"
LLC="${LLVM_PATH}/bin/llc"
CLANG="/usr/bin/clang"

INPUT_FILE=$1
DIR_NAME=$(dirname "$INPUT_FILE")
BASE_NAME=$(basename "$INPUT_FILE" .mlir)
OUTPUT_DIR="$DIR_NAME"

echo "Processing $INPUT_FILE..."

# MLIR 优化和转换流程
"$MLIR_OPT" $INPUT_FILE \
    --convert-scf-to-cf \
    --convert-cf-to-llvm \
    --convert-func-to-llvm \
    --convert-arith-to-llvm \
    --finalize-memref-to-llvm \
    --reconcile-unrealized-casts \
    -o "${OUTPUT_DIR}/${BASE_NAME}.llvm.mlir"

if [ $? -ne 0 ]; then
    echo "Error: MLIR optimization failed"
    exit 1
fi

# 转换为 LLVM IR
"$MLIR_TRANSLATE" --mlir-to-llvmir "${OUTPUT_DIR}/${BASE_NAME}.llvm.mlir" -o "${OUTPUT_DIR}/${BASE_NAME}.ll"

if [ $? -ne 0 ]; then
    echo "Error: MLIR to LLVM IR translation failed"
    exit 1
fi

# 编译为可执行文件
"$LLC" "${OUTPUT_DIR}/${BASE_NAME}.ll" -o "${OUTPUT_DIR}/${BASE_NAME}.s"
if [ $? -ne 0 ]; then
    echo "Error: LLC compilation failed"
    exit 1
fi

# 链接运行时库
"$CLANG" "${OUTPUT_DIR}/${BASE_NAME}.s" \
    -L"${LLVM_PATH}/lib" \
    -Wl,-rpath,"${LLVM_PATH}/lib" \
    -lmlir_runner_utils \
    -lmlir_c_runner_utils \
    -lmlir_float16_utils \
    -o "${OUTPUT_DIR}/${BASE_NAME}"

if [ $? -ne 0 ]; then
    echo "Error: Clang compilation failed"
    exit 1
fi

# 执行前设置库路径
export DYLD_LIBRARY_PATH="${LLVM_PATH}/lib:$DYLD_LIBRARY_PATH"

# 执行
echo "Running ${OUTPUT_DIR}/${BASE_NAME}..."
"${OUTPUT_DIR}/${BASE_NAME}"