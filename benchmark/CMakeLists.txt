cmake_minimum_required(VERSION 3.10)
project(matrix_benchmark)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加优化选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

add_executable(cpp_bench src/cpp_bench.cpp)

# 在 Mac 上链接 Accelerate Framework
if(APPLE)
    target_link_libraries(cpp_bench "-framework Accelerate")
endif()
