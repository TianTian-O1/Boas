cmake_minimum_required(VERSION 3.10)
project(MatrixCompiler)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加 LLVM 支持
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# 设置 MLIR_DIR
set(MLIR_DIR ${LLVM_DIR}/../mlir)
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

# 添加头文件路径
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
add_definitions(${LLVM_DEFINITIONS})

# 添加调试选项
option(ENABLE_DEBUG "Enable debug output" ON)
set(DEBUG_LEVEL "0" CACHE STRING "Debug level (0-31)")

if(ENABLE_DEBUG)
    add_definitions(-DDEBUG_MODE)
    add_definitions(-DDEBUG_LEVEL=${DEBUG_LEVEL})
endif()

# 添加源文件
add_library(matrix-lib
    lib/frontend/Lexer.cpp
    lib/frontend/Parser.cpp
    lib/mlirops/MLIRGen.cpp
    lib/mlirops/MLIRToC.cpp
    lib/Debug.cpp
)

# 获取 LLVM 和 MLIR 组件列表
llvm_map_components_to_libnames(llvm_libs 
    core 
    support 
    irreader
    analysis
    passes
    target
    executionengine
    native
    interpreter
    mcjit
)

# 在 find_package(MLIR REQUIRED CONFIG) 之后添加
link_directories(${LLVM_LIBRARY_DIR})
link_directories(${LLVM_INSTALL_PREFIX}/lib)

# 设置库路径
link_directories(${LLVM_INSTALL_PREFIX}/lib)

# 修改 MLIR_LIBS 列表，添加必要的方言库
set(MLIR_LIBS
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRSupport
    MLIRDialect
    MLIRExecutionEngine
    MLIRTransforms
    MLIRArithDialect
    MLIRMemRefDialect
    MLIRFuncDialect
    MLIRLinalgDialect    # 添加 Linalg 方言支持
    MLIRTensorDialect    # 添加 Tensor 方言支持
    MLIRSCFDialect
    MLIRLinalgTransforms
    MLIRTensorTransforms
)

# 添加 MLIR 方言的注册
target_compile_definitions(matrix-lib 
    PRIVATE
    MLIR_DIALECT_LINALG_ENABLE
    MLIR_DIALECT_TENSOR_ENABLE
)

# 修改链接配置
target_link_libraries(matrix-lib 
    PUBLIC 
    ${llvm_libs}
    ${MLIR_LIBS}
    Threads::Threads
)

# 添加测试可执行文件
add_executable(test-lexer test/test_Lexer.cpp)
target_link_libraries(test-lexer PRIVATE matrix-lib)

add_executable(test-parser test/test_parser.cpp)
target_link_libraries(test-parser PRIVATE matrix-lib)


# 查找 llc 和 clang
find_program(LLC_PATH NAMES llc HINTS ${LLVM_TOOLS_BINARY_DIR})
find_program(CLANG_PATH NAMES clang)

message(STATUS "Found LLC at: ${LLC_PATH}")
message(STATUS "Found Clang at: ${CLANG_PATH}")



# 在添加 test-llvm 目标之前
add_executable(matrix-compiler test/test_llvm.cpp)
target_compile_definitions(matrix-compiler 
    PRIVATE
    MLIR_OPT_PATH="${MLIR_OPT_PATH}"
    MLIR_TRANSLATE_PATH="${MLIR_TRANSLATE_PATH}"
    LLC_PATH="${LLC_PATH}"
    CLANG_PATH="${CLANG_PATH}"
    LLVM_INSTALL_PATH="${LLVM_INSTALL_PREFIX}"
)
target_link_libraries(matrix-compiler
    PRIVATE 
    matrix-lib
    ${MLIR_LIBS}
)




find_package(Threads REQUIRED)

# 在 find_package(LLVM REQUIRED CONFIG) 之后添加
if(NOT DEFINED LLVM_INSTALL_PREFIX)
    message(FATAL_ERROR "LLVM_INSTALL_PREFIX is not defined")
endif()
message(STATUS "LLVM installation prefix: ${LLVM_INSTALL_PREFIX}")

# 在 CMakeLists.txt 中添加以下内容（放在靠前的位置）

# 设置 LLVM 安装路径
if(NOT DEFINED LLVM_INSTALL_PREFIX)
    message(FATAL_ERROR "LLVM_INSTALL_PREFIX is not defined")
endif()

# 设置工具路径
set(MLIR_OPT_PATH "${LLVM_INSTALL_PREFIX}/bin/mlir-opt" CACHE PATH "Path to mlir-opt")
set(MLIR_TRANSLATE_PATH "${LLVM_INSTALL_PREFIX}/bin/mlir-translate" CACHE PATH "Path to mlir-translate")
set(LLC_PATH "${LLVM_INSTALL_PREFIX}/bin/llc" CACHE PATH "Path to llc")
set(CLANG_PATH "/usr/bin/clang" CACHE PATH "Path to clang")

# 为所有目标添加这些定义
add_definitions(
    -DMLIR_OPT_PATH="${MLIR_OPT_PATH}"
    -DMLIR_TRANSLATE_PATH="${MLIR_TRANSLATE_PATH}"
    -DLLC_PATH="${LLC_PATH}"
    -DCLANG_PATH="${CLANG_PATH}"
    -DLLVM_INSTALL_PATH="${LLVM_INSTALL_PREFIX}"
)

# 确保这些工具存在
find_program(MLIR_OPT_EXECUTABLE NAMES mlir-opt HINTS ${LLVM_INSTALL_PREFIX}/bin)
find_program(MLIR_TRANSLATE_EXECUTABLE NAMES mlir-translate HINTS ${LLVM_INSTALL_PREFIX}/bin)
find_program(LLC_EXECUTABLE NAMES llc HINTS ${LLVM_INSTALL_PREFIX}/bin)
find_program(CLANG_EXECUTABLE NAMES clang)

if(NOT MLIR_OPT_EXECUTABLE)
    message(FATAL_ERROR "mlir-opt not found")
endif()
if(NOT MLIR_TRANSLATE_EXECUTABLE)
    message(FATAL_ERROR "mlir-translate not found")
endif()
if(NOT LLC_EXECUTABLE)
    message(FATAL_ERROR "llc not found")
endif()
if(NOT CLANG_EXECUTABLE)
    message(FATAL_ERROR "clang not found")
endif()

message(STATUS "Found mlir-opt: ${MLIR_OPT_EXECUTABLE}")
message(STATUS "Found mlir-translate: ${MLIR_TRANSLATE_EXECUTABLE}")
message(STATUS "Found llc: ${LLC_EXECUTABLE}")
message(STATUS "Found clang: ${CLANG_EXECUTABLE}")


# 修改 matrix-lib 的编译定义
target_compile_definitions(matrix-lib 
    PRIVATE
    MLIR_DIALECT_LINALG_ENABLE
    MLIR_DIALECT_TENSOR_ENABLE
    MLIR_OPT_PATH="${MLIR_OPT_PATH}"
    MLIR_TRANSLATE_PATH="${MLIR_TRANSLATE_PATH}"
    LLC_PATH="${LLC_PATH}"
    CLANG_PATH="${CLANG_PATH}"
    LLVM_INSTALL_PATH="${LLVM_INSTALL_PREFIX}"
)