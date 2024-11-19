#!/bin/bash

# 定义颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 定义调试级别
DEBUG_NONE=0
DEBUG_LEXER=1
DEBUG_PARSER=2
DEBUG_AST=4
DEBUG_MLIR=8
DEBUG_LLVM=16
DEBUG_ALL=31

# 默认调试级别
DEBUG_LEVEL=$DEBUG_NONE

# 解析命令行参数
while getopts "d:t:h" opt; do
  case $opt in
    d)
      case $OPTARG in
        none) DEBUG_LEVEL=$DEBUG_NONE ;;
        lexer) DEBUG_LEVEL=$DEBUG_LEXER ;;
        parser) DEBUG_LEVEL=$DEBUG_PARSER ;;
        ast) DEBUG_LEVEL=$DEBUG_AST ;;
        mlir) DEBUG_LEVEL=$DEBUG_MLIR ;;
        llvm) DEBUG_LEVEL=$DEBUG_LLVM ;;
        all) DEBUG_LEVEL=$DEBUG_ALL ;;
        *) echo "Invalid debug level"; exit 1 ;;
      esac
      ;;
    t)
      if [ -f "$OPTARG" ]; then
        TEST_FILE="$OPTARG"
      else
        echo "Error: Test file '$OPTARG' does not exist"
        exit 1
      fi
      ;;
    h)
      echo "Usage: $0 [-d debug_level] [-t test_file]"
      echo "Debug levels: none, lexer, parser, ast, mlir, llvm, all"
      echo "Example: $0 -d parser -t test/matmul.bs"
      exit 0
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

# 清理和创建构建目录
echo -e "${YELLOW}Cleaning build directory...${NC}"
rm -rf build
mkdir build
cd build

# 配置 CMake
echo -e "${YELLOW}Configuring CMake...${NC}"
cmake -DLLVM_DIR=/Users/mac/llvm-install/lib/cmake/llvm \
      -DENABLE_DEBUG=ON \
      -DDEBUG_LEVEL=$DEBUG_LEVEL ..

# 构建项目
echo -e "${YELLOW}Building project...${NC}"
make

# 如果构建成功且指定了测试文件，则运行测试
# 修改后的测试运行部分
if [ $? -eq 0 ] && [ ! -z "$TEST_FILE" ]; then
    echo -e "${GREEN}Build successful! Running tests...${NC}"
    
    # 获取完整的测试文件路径
    TEST_FILE_PATH="$TEST_FILE"
    if [[ ! "$TEST_FILE" = /* ]]; then
        TEST_FILE_PATH="$(pwd)/../$TEST_FILE"
    fi
    
    echo -e "${YELLOW}Running lexer test...${NC}"
    ./test-lexer "$TEST_FILE_PATH"
    
    echo -e "${YELLOW}Running parser test...${NC}"
    ./test-parser "$TEST_FILE_PATH"
    
    echo -e "${YELLOW}Running LLVM test...${NC}"
    ./test-llvm "$TEST_FILE_PATH"
else
    echo -e "${RED}Build failed or no test file specified${NC}"
fi