find_package(GTest REQUIRED)
find_package(Python3 REQUIRED)

# 添加 Python 测试
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/test_list.py ${CMAKE_CURRENT_BINARY_DIR}/test_list.py COPYONLY)

# C++ 测试可执行文件
add_executable(list_test
    unit/list_file.cpp
)

target_link_libraries(list_test
    PRIVATE
    boas
    GTest::GTest
    GTest::Main
    MLIRIR
    MLIRSupport
    MLIRParser
    MLIRPass
    MLIRTransforms
    MLIRLLVMDialect
    MLIRLLVMToLLVMIRTranslation
    MLIRLLVMIRTransforms
    MLIRExecutionEngine
    MLIRJitRunner
    MLIRTargetLLVMIRExport
    MLIRAnalysis
    MLIRDialect
    MLIRAsmParser
    MLIRTranslateLib
)

target_include_directories(list_test
    PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${MLIR_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS}
)

# 添加 Python 测试
add_test(NAME python_list_test 
         COMMAND Python3::Interpreter ${CMAKE_CURRENT_BINARY_DIR}/test_list.py
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# C++ 测试
add_test(NAME cpp_list_test COMMAND list_test)
