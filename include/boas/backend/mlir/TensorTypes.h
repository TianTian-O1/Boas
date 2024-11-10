#ifndef BOAS_TENSOR_TYPES_H
#define BOAS_TENSOR_TYPES_H

#include "mlir/IR/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace boas {
namespace mlir {

class TensorType : public ::mlir::Type::TypeBase<TensorType, ::mlir::Type, 
                                                ::mlir::TypeStorage> {
public:
    // Required type name
    static constexpr ::llvm::StringRef name = "boas.tensor";

    // Static construction methods
    static TensorType get(::mlir::Type elementType,
                         ::llvm::ArrayRef<int64_t> shape,
                         ::mlir::MLIRContext* context);

    // Shape accessor
    ::llvm::ArrayRef<int64_t> getShape() const;
    ::mlir::Type getElementType() const;

private:
    using Base::Base;
};

} // namespace mlir
} // namespace boas

#endif // BOAS_TENSOR_TYPES_H
