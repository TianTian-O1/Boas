add_mlir_dialect_library(MLIRBoasDialect
  BoasDialect.cpp
  BoasOps.cpp
  BoasTypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Boas/Dialect/Boas

  DEPENDS
  MLIRBoasOpsIncGen
  MLIRBoasTypesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
)
