get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(boas-run
  boas-run.cpp
)

llvm_update_compile_flags(boas-run)

target_link_libraries(boas-run
  PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRSupport
  MLIRExecutionEngine
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRTensorDialect
  MLIRArithDialect
  MLIRArithTransforms
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRControlFlowDialect
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRReconcileUnrealizedCasts
  MLIRTargetLLVMIRExport
  MLIRToLLVMIRTranslationRegistration
  MLIRArithToLLVM
  MLIRFuncToLLVM
  MLIRMemRefToLLVM
  MLIRSCFToControlFlow
  LLVMSupport
  LLVMCore
)
