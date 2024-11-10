#ifndef BOAS_MLIR_TENSOR_TYPES_H
#define BOAS_MLIR_TENSOR_TYPES_H

#include "mlir/IR/Types.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/DialectImplementation.h"

namespace boas {
namespace mlir {

namespace detail {
struct TensorTypeStorage : public ::mlir::TypeStorage {
    using KeyTy = std::tuple<::mlir::Type, std::vector<int64_t>>;
    
    TensorTypeStorage(::mlir::Type elementType, std::vector<int64_t> shape)
        : elementType(elementType), shape(shape) {}
    
    bool operator==(const KeyTy &key) const {
        return elementType == std::get<0>(key) && shape == std::get<1>(key);
    }
    
    ::mlir::Type elementType;
    std::vector<int64_t> shape;
};
} // namespace detail

class TensorType : public ::mlir::Type::TypeBase<TensorType, ::mlir::Type,
                                              detail::TensorTypeStorage> {
public:
    using Base::Base;
    
    static TensorType get(::mlir::Type elementType, 
                         ArrayRef<int64_t> shape,
                         ::mlir::MLIRContext *context);
    
    ::mlir::Type getElementType() const;
    ArrayRef<int64_t> getShape() const;
};

} // namespace mlir
} // namespace boas

#endif
