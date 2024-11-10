#ifndef BOAS_MLIR_LIST_TYPES_H
#define BOAS_MLIR_LIST_TYPES_H

#include "mlir/IR/Types.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/DialectImplementation.h"

namespace boas {
namespace mlir {

namespace detail {
struct ListTypeStorage : public ::mlir::TypeStorage {
    using KeyTy = std::tuple<>;
    
    static ListTypeStorage *construct(::mlir::TypeStorageAllocator &allocator,
                                    const KeyTy &) {
        return new (allocator.allocate<ListTypeStorage>()) ListTypeStorage();
    }
    
    bool operator==(const KeyTy &) const { return true; }
};
} // namespace detail

class ListType : public ::mlir::Type::TypeBase<ListType, ::mlir::Type,
                                             detail::ListTypeStorage> {
public:
    using Base::Base;
    
    static constexpr const char *name = "boas.list";
    
    static ListType get(::mlir::MLIRContext *context) {
        return Base::get(context);
    }
    
    static ::mlir::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
        return ::mlir::success();
    }
    
    static ::mlir::TypeID getTypeID() {
        return ::mlir::TypeID::get<ListType>();
    }
};

} // namespace mlir
} // namespace boas

#endif // BOAS_MLIR_LIST_TYPES_H
