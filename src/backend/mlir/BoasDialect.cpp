#include "boas/backend/mlir/BoasDialect.h"
#include "boas/backend/mlir/ListOps.h"
#include "boas/backend/mlir/NumberOps.h"
#include "boas/backend/mlir/PrintOps.h"
#include "boas/backend/mlir/TensorOps.h"

namespace boas {
namespace mlir {

BoasDialect::BoasDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<BoasDialect>()) {
    addOperations<ListCreateOp, ListAppendOp, ListSliceOp, 
                 ListNestedCreateOp, ListGetOp,
                 TensorCreateOp, TensorMatMulOp,
                 NumberConstantOp, PrintOp>();
}

void BoasDialect::initialize() {
    addTypes<ListType, TensorType>();
}

} // namespace mlir
} // namespace boas
