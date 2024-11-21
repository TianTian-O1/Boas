cmake_minimum_required(VERSION 3.10)
project(matrix_benchmark)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加优化选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# 设置正确的库搜索路径 (使用绝对路径)
get_filename_component(PROJECT_ROOT "${CMAKE_SOURCE_DIR}/.." ABSOLUTE)
link_directories("${PROJECT_ROOT}/build/lib")

# 添加包含目录
include_directories("${PROJECT_ROOT}/include")

add_executable(cpp_bench src/cpp_bench.cpp)

# 链接 matrix-runtime 库
target_link_libraries(cpp_bench matrix-runtime)

# 在 Mac 上链接 Accelerate Framework
if(APPLE)
    target_link_libraries(cpp_bench "-framework Accelerate")
endif()

# 设置运行时路径
set_target_properties(cpp_bench PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "${PROJECT_ROOT}/build/lib"
)
