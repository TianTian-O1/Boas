#ifndef BOAS_MLIR_LIST_TYPES_H
#define BOAS_MLIR_LIST_TYPES_H

#include "mlir/IR/Types.h"
#include "mlir/IR/TypeSupport.h"

namespace boas {
namespace mlir {

class ListType : public ::mlir::Type::TypeBase<ListType, ::mlir::Type,
                                             ::mlir::TypeStorage> {
public:
    using Base::Base;
    
    static constexpr const char *name = "boas.list";
    
    static ListType get(::mlir::MLIRContext *context) {
        return Base::get(context);
    }
    
    static ::mlir::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()>) {
        return ::mlir::success();
    }
    
    static ::mlir::TypeID getTypeID() {
        return ::mlir::TypeID::get<ListType>();
    }
};

} // namespace mlir
} // namespace boas

#endif // BOAS_MLIR_LIST_TYPES_H
