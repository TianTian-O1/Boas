cmake_minimum_required(VERSION 3.10)
project(MatrixCompiler)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置 LLVM 安装路径
if(NOT DEFINED LLVM_INSTALL_PREFIX)
    set(LLVM_INSTALL_PREFIX "/user/local" CACHE PATH "LLVM installation directory")
endif()

message(STATUS "LLVM installation prefix: ${LLVM_INSTALL_PREFIX}")

# 添加 LLVM 支持
list(APPEND CMAKE_PREFIX_PATH "${LLVM_INSTALL_PREFIX}/lib/cmake/llvm")
find_package(LLVM REQUIRED CONFIG)
list(APPEND CMAKE_PREFIX_PATH "${LLVM_INSTALL_PREFIX}/lib/cmake/mlir")
find_package(MLIR REQUIRED CONFIG)

# 设置工具路径
set(MLIR_OPT "${LLVM_INSTALL_PREFIX}/bin/mlir-opt")
set(MLIR_TRANSLATE "${LLVM_INSTALL_PREFIX}/bin/mlir-translate")
set(LLC "${LLVM_INSTALL_PREFIX}/bin/llc")
set(CLANG "/usr/bin/clang")

# 添加包含目录
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)

# 验证工具是否存在
foreach(tool ${MLIR_OPT} ${MLIR_TRANSLATE} ${LLC})
    if(NOT EXISTS ${tool})
        message(FATAL_ERROR "Required tool not found: ${tool}")
    endif()
endforeach()

# 设置编译定义
add_definitions(
    -DMLIR_OPT_PATH="${MLIR_OPT}"
    -DMLIR_TRANSLATE_PATH="${MLIR_TRANSLATE}"
    -DLLC_PATH="${LLC}"
    -DCLANG_PATH="${CLANG}"
    -DLLVM_INSTALL_PATH="${LLVM_INSTALL_PREFIX}"
    -DPROJECT_ROOT="${CMAKE_SOURCE_DIR}"
)

# 创建运行时库
add_library(matrix-runtime SHARED
    lib/runtime/PrintUtils.cpp
)

# 设置运行时库的输出目录
set_target_properties(matrix-runtime PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)


# 添加源文件
add_library(matrix-lib
    lib/frontend/Lexer.cpp
    lib/frontend/Parser.cpp
    lib/frontend/PythonASTParser.cpp
    lib/frontend/python_parser/PythonASTBuilder.cpp
    lib/frontend/python_parser/PythonASTVisitor.cpp
    lib/frontend/python_parser/PythonASTNodes.cpp
    lib/mlirops/MLIRGen.cpp
    lib/mlirops/MLIRGenMatrix.cpp
    lib/mlirops/MLIRGenNodes.cpp
    lib/mlirops/MLIRGenTensor.cpp
    lib/mlirops/MLIRGenTiming.cpp
    lib/mlirops/MLIRGenUtils.cpp
    lib/mlirops/MLIRGenOptimization.cpp
    lib/mlirops/TensorMemoryPool.cpp
    lib/Debug.cpp
)

# 获取 LLVM 和 MLIR 库
llvm_map_components_to_libnames(llvm_libs 
    core 
    support 
    irreader
    analysis
)

set(MLIR_LIBS
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRSupport
    MLIRTransforms
    MLIRLinalgDialect
    MLIRTensorDialect
    MLIRVectorDialect           # 添加 Vector dialect 支持
    MLIRVectorToSCF            # Vector 到 SCF 的转换
    MLIRSCFDialect             # SCF dialect 支持
    MLIRArithDialect           # Arith dialect 支持
    MLIRMemRefDialect          # MemRef dialect 支持
    MLIRLinalgTransforms       # Linalg 变换支持
)

# 设置库的编译选项
target_compile_definitions(matrix-lib PRIVATE
    MLIR_OPT_PATH="${MLIR_OPT}"
    MLIR_TRANSLATE_PATH="${MLIR_TRANSLATE}"
    LLC_PATH="${LLC}"
    CLANG_PATH="${CLANG}"
    LLVM_INSTALL_PATH="${LLVM_INSTALL_PREFIX}"
)

target_link_libraries(matrix-lib PUBLIC ${llvm_libs} ${MLIR_LIBS})

# 添加可执行文件
add_executable(matrix-compiler test/test_full_pipeline.cpp)
target_compile_definitions(matrix-compiler PRIVATE
    MLIR_OPT_PATH="${MLIR_OPT}"
    MLIR_TRANSLATE_PATH="${MLIR_TRANSLATE}"
    LLC_PATH="${LLC}"
    CLANG_PATH="${CLANG}"
    LLVM_INSTALL_PATH="${LLVM_INSTALL_PREFIX}"
)
target_link_libraries(matrix-compiler PRIVATE matrix-lib)


# 添加Python支持
find_package(Python3 COMPONENTS Development REQUIRED)
include_directories(${Python3_INCLUDE_DIRS})

# 在现有的target_link_libraries中添加Python库
target_link_libraries(matrix-compiler
    PRIVATE
    ${Python3_LIBRARIES}
    # ... 其他库
)

# 添加Python AST测试
add_executable(test-python-ast test/test_python_ast.cpp lib/frontend/PythonAST.cpp)
target_link_libraries(test-python-ast 
    PRIVATE 
    ${Python3_LIBRARIES}
    matrix-lib
)

# 添加 Python AST 解析器测试
add_executable(test-python-parser test/test_python_parser.cpp)
target_link_libraries(test-python-parser 
    PRIVATE 
    matrix-lib
    ${Python3_LIBRARIES}
)

# 添加完整流程测试
add_executable(test-full-pipeline test/test_full_pipeline.cpp)
target_link_libraries(test-full-pipeline 
    PRIVATE 
    matrix-lib
    ${Python3_LIBRARIES}
)

# Debug 输出 paths
message(STATUS "MLIR_OPT path: ${MLIR_OPT}")
message(STATUS "MLIR_TRANSLATE path: ${MLIR_TRANSLATE}")
message(STATUS "LLC path: ${LLC}")
message(STATUS "CLANG path: ${CLANG}")