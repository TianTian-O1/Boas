#ifndef BOAS_MLIR_TENSOR_OPS_H
#define BOAS_MLIR_TENSOR_OPS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "boas/backend/mlir/TensorTypes.h"
#include "llvm/ADT/ArrayRef.h"

namespace boas {
namespace mlir {

class TensorCreateOp : public ::mlir::Op<TensorCreateOp,
                                      ::mlir::OpTrait::ZeroOperands,
                                      ::mlir::OpTrait::OneResult> {
public:
    using Op::Op;
    
    static void build(::mlir::OpBuilder &builder, ::mlir::OperationState &state,
                     ::mlir::Type elementType, ::llvm::ArrayRef<int64_t> shape);
                     
    ::mlir::Value getResult() { return this->getOperation()->getResult(0); }
    
    static ::llvm::StringRef getOperationName() { return "boas.tensor.create"; }

    static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() { 
        static ::llvm::StringRef names[] = {"shape"};
        return ::llvm::ArrayRef<::llvm::StringRef>(names);
    }

    ::llvm::ArrayRef<int64_t> getShape() {
        auto shapeAttr = this->getOperation()->getAttrOfType<::mlir::DenseI64ArrayAttr>("shape");
        return shapeAttr.asArrayRef();
    }
};

class TensorMatMulOp : public ::mlir::Op<TensorMatMulOp,
                                      ::mlir::OpTrait::NOperands<2>::Impl,
                                      ::mlir::OpTrait::OneResult> {
public:
    using Op::Op;
    
    static void build(::mlir::OpBuilder &builder, ::mlir::OperationState &state,
                     ::mlir::Value lhs, ::mlir::Value rhs);
                     
    ::mlir::Value getResult() { return this->getOperation()->getResult(0); }
    
    static ::llvm::StringRef getOperationName() { return "boas.tensor.matmul"; }

    static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
        return {};
    }
};

} // namespace mlir
} // namespace boas

#endif
