#pragma once
#include "mlir/IR/Dialect.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace boas {

class BoasDialect : public Dialect {
public:
  explicit BoasDialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "boas"; }
};

} // namespace boas
} // namespace mlir