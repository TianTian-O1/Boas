#include "boas/backend/mlir/BoasDialect.h"
#include "boas/backend/mlir/ListOps.h"
#include "boas/backend/mlir/NumberOps.h"
#include "boas/backend/mlir/PrintOps.h"

namespace boas {
namespace mlir {

BoasDialect::BoasDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<BoasDialect>()) {
    addOperations<ListCreateOp, ListAppendOp, ListSliceOp, 
                 ListNestedCreateOp, ListGetOp,
                 NumberConstantOp, PrintOp>();
}

void BoasDialect::initialize() {
    addTypes<ListType>();
}

} // namespace mlir
} // namespace boas
