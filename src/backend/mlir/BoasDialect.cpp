#include "boas/backend/mlir/BoasDialect.h"
#include "boas/backend/mlir/ListOps.h"
#include "boas/backend/mlir/ListTypes.h"
#include "boas/backend/mlir/PrintOps.h"
#include "boas/backend/mlir/NumberOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

namespace boas {
namespace mlir {

BoasDialect::BoasDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<BoasDialect>()) {
    addOperations<ListCreateOp, ListAppendOp, ListSliceOp, 
                 ListNestedCreateOp, ListGetOp,
                 ListComprehensionOp,
                 NumberConstantOp, PrintOp>();
                 
    // Register the ListType using the non-template version
    addType(ListType::getTypeID(), 
            ::mlir::AbstractType::get<ListType>(*this));
}

} // namespace mlir
} // namespace boas
