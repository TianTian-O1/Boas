add_mlir_library(MLIRBoasToLinalg
  BoasToLinalg.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Boas/Conversion

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRLinalgDialect
  MLIRTensorDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRTransforms
)
