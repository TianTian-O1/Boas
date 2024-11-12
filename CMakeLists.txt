cmake_minimum_required(VERSION 3.10)
project(MatrixCompiler)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加 LLVM 支持
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# 添加 LLVM 头文件路径
include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

# 添加包含目录
include_directories(${PROJECT_SOURCE_DIR}/include)

# 添加调试选项
option(ENABLE_DEBUG "Enable debug output" ON)
set(DEBUG_LEVEL "0" CACHE STRING "Debug level (0-31)")

# 配置调试宏
if(ENABLE_DEBUG)
    add_definitions(-DDEBUG_MODE)
    add_definitions(-DDEBUG_LEVEL=${DEBUG_LEVEL})
endif()

# 添加新的源文件
add_library(matrix-lib
    lib/frontend/Lexer.cpp
    lib/frontend/Parser.cpp
    lib/mlirops/MLIRGen.cpp
    lib/mlirops/MLIRToLLVM.cpp
    lib/Debug.cpp
)

# 获取 LLVM 组件列表
llvm_map_components_to_libnames(llvm_libs 
    core 
    support 
    irreader
    analysis
    passes
    target
)

# 链接 LLVM 库到我们的目标
target_link_libraries(matrix-lib PRIVATE ${llvm_libs})

# 添加测试可执行文件
add_executable(test-lexer test/test_Lexer.cpp)
target_link_libraries(test-lexer matrix-lib)

add_executable(test-parser test/test_parse.cpp)
target_link_libraries(test-parser matrix-lib)

# 查找 llc 和 clang
find_program(LLC_PATH NAMES llc HINTS ${LLVM_TOOLS_BINARY_DIR})
find_program(CLANG_PATH NAMES clang)

message(STATUS "Found LLC at: ${LLC_PATH}")
message(STATUS "Found Clang at: ${CLANG_PATH}")

add_executable(test-llvm test/test_llvm.cpp)
target_compile_definitions(test-llvm PRIVATE
    LLC_PATH="${LLC_PATH}"
    CLANG_PATH="${CLANG_PATH}"
    DEBUG_LEVEL=${DEBUG_LEVEL}
)
target_link_libraries(test-llvm matrix-lib ${llvm_libs})
