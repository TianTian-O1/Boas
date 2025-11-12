get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(boas-opt
  boas-opt.cpp

  DEPENDS
  MLIRBoasToLinalg
)

llvm_update_compile_flags(boas-opt)

target_link_libraries(boas-opt
  PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  MLIRBoasToLinalg
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRSupport
  MLIROptLib
  MLIRLinalgDialect
  MLIRTensorDialect
  MLIRArithDialect
  MLIRFuncDialect
)
