get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(boas-compile
  boas-compile.cpp
)

llvm_update_compile_flags(boas-compile)

target_link_libraries(boas-compile
  PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRSupport
  MLIRLinalgDialect
  MLIRTensorDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRSupport
  LLVMSupport
)
