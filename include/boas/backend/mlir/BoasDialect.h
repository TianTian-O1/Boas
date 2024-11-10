#ifndef BOAS_MLIR_DIALECT_H
#define BOAS_MLIR_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

namespace boas {
namespace mlir {

class BoasDialect : public ::mlir::Dialect {
public:
    explicit BoasDialect(::mlir::MLIRContext *context);
    static ::llvm::StringRef getDialectNamespace() { return "boas"; }

    void initialize();
};

} // namespace mlir
} // namespace boas

#endif // BOAS_MLIR_DIALECT_H
