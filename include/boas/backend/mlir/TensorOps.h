#ifndef BOAS_MLIR_TENSOR_OPS_H
#define BOAS_MLIR_TENSOR_OPS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "boas/backend/mlir/TensorTypes.h"

namespace boas {
namespace mlir {

class TensorCreateOp : public ::mlir::Op<TensorCreateOp,
                                      ::mlir::OpTrait::ZeroOperands,
                                      ::mlir::OpTrait::OneResult> {
public:
    using Op::Op;
    
    static void build(::mlir::OpBuilder &builder, ::mlir::OperationState &state,
                     ::mlir::Type elementType, ArrayRef<int64_t> shape);
                     
    ::mlir::Value getResult() { return this->getOperation()->getResult(0); }
    
    static ::llvm::StringRef getOperationName() { return "boas.tensor.create"; }
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
};

} // namespace mlir
} // namespace boas

#endif
