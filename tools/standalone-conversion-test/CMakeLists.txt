add_llvm_executable(standalone-matmul-conversion
  StandaloneMatMulConversion.cpp
)

llvm_update_compile_flags(standalone-matmul-conversion)

target_link_libraries(standalone-matmul-conversion
  PRIVATE
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRLinalgDialect
  MLIRTensorDialect
  MLIRArithDialect
  MLIRFuncDialect
)
