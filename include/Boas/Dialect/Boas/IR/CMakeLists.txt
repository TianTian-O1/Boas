# Generate Ops and Dialect code
set(LLVM_TARGET_DEFINITIONS BoasOps.td)
mlir_tablegen(BoasOps.h.inc -gen-op-decls)
mlir_tablegen(BoasOps.cpp.inc -gen-op-defs)
mlir_tablegen(BoasDialect.h.inc -gen-dialect-decls -dialect=boas)
mlir_tablegen(BoasDialect.cpp.inc -gen-dialect-defs -dialect=boas)
add_public_tablegen_target(MLIRBoasOpsIncGen)

# Generate Types code
set(LLVM_TARGET_DEFINITIONS BoasTypes.td)
mlir_tablegen(BoasTypes.h.inc -gen-typedef-decls -typedefs-dialect=boas)
mlir_tablegen(BoasTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=boas)
add_public_tablegen_target(MLIRBoasTypesIncGen)

# Documentation
add_mlir_doc(BoasDialect BoasDialect Boas/ -gen-dialect-doc)
add_mlir_doc(BoasOps BoasOps Boas/ -gen-op-doc)
