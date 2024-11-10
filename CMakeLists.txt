cmake_minimum_required(VERSION 3.10)
project(boas-lang)

# Set policy for Python modules
cmake_policy(SET CMP0148 NEW)

# Use newer FindPython module
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Python configuration
include_directories(${Python_INCLUDE_DIRS})
link_directories(${Python_LIBRARY_DIRS})

# LLVM configuration
find_package(LLVM REQUIRED CONFIG)
include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

# MLIR configuration
find_package(MLIR REQUIRED CONFIG)
include_directories(${MLIR_INCLUDE_DIRS})

# Add subdirectories
add_subdirectory(src)
add_subdirectory(tests)

# Enable testing
enable_testing()