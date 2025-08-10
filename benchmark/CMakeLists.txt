cmake_minimum_required(VERSION 3.10)
project(matrix_benchmark)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加优化选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# 设置包含目录
include_directories(${PROJECT_SOURCE_DIR}/include)

# 添加可执行文件
add_executable(cpp_bench src/cpp_bench.cpp)
add_executable(perf_analysis src/perf_analysis.cpp)

# 在 Mac 上链接 Accelerate Framework
if(APPLE)
    target_link_libraries(cpp_bench "-framework Accelerate")
    target_link_libraries(perf_analysis "-framework Accelerate")
endif()

# 添加文件系统库支持
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    link_libraries(stdc++fs)
endif()
