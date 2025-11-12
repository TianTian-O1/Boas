cmake_minimum_required(VERSION 3.28)
project(Boas-NPU LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============================================================================
# Options
# ============================================================================
option(BOAS_ENABLE_NPU "Enable NPU support" ON)
option(BOAS_ENABLE_PYTHON_BINDINGS "Enable Python bindings" ON)
option(BOAS_ENABLE_TESTING "Enable testing" ON)

# ============================================================================
# Find LLVM/MLIR
# ============================================================================
# 使用已编译的AscendNPU-IR中的LLVM/MLIR
set(AscendNPU_IR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/AscendNPU-IR"
    CACHE PATH "AscendNPU-IR source directory")
set(AscendNPU_IR_BUILD_DIR "${AscendNPU_IR_DIR}/build"
    CACHE PATH "AscendNPU-IR build directory")

# LLVM/MLIR路径
list(APPEND CMAKE_PREFIX_PATH "${AscendNPU_IR_BUILD_DIR}/lib/cmake/llvm")
list(APPEND CMAKE_PREFIX_PATH "${AscendNPU_IR_BUILD_DIR}/lib/cmake/mlir")

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

# ============================================================================
# Include directories
# ============================================================================
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

# BiShengIR includes
include_directories(${AscendNPU_IR_DIR}/bishengir/include)
include_directories(${AscendNPU_IR_BUILD_DIR}/bishengir/include)

# ============================================================================
# LLVM/MLIR definitions
# ============================================================================
add_definitions(${LLVM_DEFINITIONS})
add_definitions(${MLIR_DEFINITIONS})

# ============================================================================
# LLVM/MLIR libraries
# ============================================================================
llvm_map_components_to_libnames(llvm_libs
  Core
  Support
  Analysis
  IRReader
  TransformUtils
)

set(MLIR_LIBS
  # MLIR Core
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIRTransformUtils
  MLIROptLib

  # MLIR Dialects
  MLIRFunc
  MLIRFuncDialect
  MLIRTensorDialect
  MLIRLinalgDialect
  MLIRArithDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRVectorDialect

  # MLIR Conversions
  MLIRLinalgTransforms
  MLIRVectorToSCF

  # MLIR ExecutionEngine (optional)
  MLIRExecutionEngine
  MLIRJitRunner
)

# BiShengIR libraries
set(BISHENGIR_LIBS
  # BiShengIR Dialects
  BiShengIRHFusionDialect
  BiShengIRHIVMDialect
  BiShengIRTensorDialect

  # BiShengIR Conversions
  BiShengIRConversionLinalgToHFusion
  BiShengIRConversionHFusionToHIVM

  # BiShengIR Transforms
  BiShengIRTransformsHFusion
)

# ============================================================================
# Find CANN (optional, for NPU support)
# ============================================================================
if(BOAS_ENABLE_NPU)
  set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
  if(NOT ASCEND_HOME_PATH)
    set(ASCEND_HOME_PATH "/usr/local/Ascend/latest")
  endif()

  message(STATUS "CANN install path: ${ASCEND_HOME_PATH}")

  # ACL includes and libraries
  include_directories(${ASCEND_HOME_PATH}/include)
  link_directories(${ASCEND_HOME_PATH}/lib64)

  set(ACL_LIBS
    ascendcl
    acl_op_compiler
  )
endif()

# ============================================================================
# TableGen
# ============================================================================
# Add MLIR CMake modules to path
list(APPEND CMAKE_MODULE_PATH "${AscendNPU_IR_BUILD_DIR}/lib/cmake/mlir")
list(APPEND CMAKE_MODULE_PATH "${AscendNPU_IR_BUILD_DIR}/lib/cmake/llvm")

# Include LLVM and MLIR CMake modules
include(TableGen)
include(AddLLVM)
include(AddMLIR)

set(MLIR_TABLEGEN_EXE mlir-tblgen)

# ============================================================================
# Subdirectories
# ============================================================================
add_subdirectory(include/Boas)
add_subdirectory(lib)
add_subdirectory(tools)

if(BOAS_ENABLE_TESTING)
  add_subdirectory(test)
endif()

if(BOAS_ENABLE_PYTHON_BINDINGS)
  add_subdirectory(python)
endif()

# ============================================================================
# Install
# ============================================================================
install(DIRECTORY include/Boas
  DESTINATION include
  FILES_MATCHING PATTERN "*.h"
)
