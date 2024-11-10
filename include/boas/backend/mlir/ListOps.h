#ifndef BOAS_MLIR_LIST_OPS_H
#define BOAS_MLIR_LIST_OPS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "boas/backend/mlir/ListTypes.h"

namespace boas {
namespace mlir {

class ListCreateOp : public ::mlir::Op<ListCreateOp,
                                     ::mlir::OpTrait::ZeroOperands,
                                     ::mlir::OpTrait::OneResult> {
public:
    using Op::Op;
    
    static void build(::mlir::OpBuilder &builder, ::mlir::OperationState &state) {
        state.addTypes(::mlir::UnrankedTensorType::get(builder.getF64Type()));
    }
    
    ::mlir::Value getResult() { return this->getOperation()->getResult(0); }
    
    static ::llvm::StringRef getOperationName() { return "boas.list.create"; }
    static ::mlir::LogicalResult verify() { return ::mlir::success(); }
    static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() { return {}; }
};

class ListAppendOp : public ::mlir::Op<ListAppendOp,
                                      ::mlir::OpTrait::OneResult,
                                      ::mlir::OpTrait::ZeroSuccessors> {
public:
    using Op::Op;
    
    static void build(::mlir::OpBuilder &builder, ::mlir::OperationState &state,
                     ::mlir::Value list, ::mlir::Value element) {
        state.addOperands({list, element});
        state.addTypes(list.getType());
    }
    
    ::mlir::Value getResult() { return this->getOperation()->getResult(0); }
    
    static ::llvm::StringRef getOperationName() { return "boas.list.append"; }
    static ::mlir::LogicalResult verify() { return ::mlir::success(); }
    static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() { return {}; }
};

class ListComprehensionOp : public ::mlir::Op<ListComprehensionOp,
                                           ::mlir::OpTrait::OneResult,
                                           ::mlir::OpTrait::ZeroSuccessors> {
public:
    using Op::Op;
    
    static void build(::mlir::OpBuilder &builder, ::mlir::OperationState &state,
                     ::mlir::Value expression, ::mlir::Value iterableList,
                     ::mlir::Value condition = nullptr) {
        state.addOperands({expression, iterableList});
        if (condition) state.addOperands(condition);
        state.addTypes(iterableList.getType());
    }
    
    static ::llvm::StringRef getOperationName() { return "boas.list.comprehension"; }
};

class ListNestedCreateOp : public ::mlir::Op<ListNestedCreateOp,
                                          ::mlir::OpTrait::OneResult,
                                          ::mlir::OpTrait::ZeroSuccessors> {
public:
    using Op::Op;
    
    static void build(::mlir::OpBuilder &builder, ::mlir::OperationState &state) {
        state.addTypes(ListType::get(builder.getContext()));
    }
    
    ::mlir::Value getResult() { return this->getOperation()->getResult(0); }
    
    static ::llvm::StringRef getOperationName() { return "boas.list.nested.create"; }
    static ::mlir::LogicalResult verify() { return ::mlir::success(); }
    static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() { return {}; }
};

class ListSliceOp : public ::mlir::Op<ListSliceOp,
                                    ::mlir::OpTrait::OneResult,
                                    ::mlir::OpTrait::ZeroSuccessors> {
public:
    using Op::Op;
    
    static void build(::mlir::OpBuilder &builder, ::mlir::OperationState &state,
                     ::mlir::Value list, ::mlir::Value start, ::mlir::Value end) {
        state.addOperands({list, start, end});
        state.addTypes(list.getType());
    }
    
    ::mlir::Value getResult() { return this->getOperation()->getResult(0); }
    
    static ::llvm::StringRef getOperationName() { return "boas.list.slice"; }
    static ::mlir::LogicalResult verify() { return ::mlir::success(); }
    static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() { return {}; }
};

class ListGetOp : public ::mlir::Op<ListGetOp,
                                   ::mlir::OpTrait::OneResult,
                                   ::mlir::OpTrait::ZeroSuccessors> {
public:
    using Op::Op;
    
    static void build(::mlir::OpBuilder &builder, ::mlir::OperationState &state,
                     ::mlir::Value list, ::mlir::Value index) {
        state.addOperands({list, index});
        state.addTypes(builder.getI64Type());
    }
    
    ::mlir::Value getList() { return this->getOperation()->getOperand(0); }
    ::mlir::Value getIndex() { return this->getOperation()->getOperand(1); }
    ::mlir::Value getResult() { return this->getOperation()->getResult(0); }
    
    static ::llvm::StringRef getOperationName() { return "boas.list.get"; }
    static ::mlir::LogicalResult verify() { return ::mlir::success(); }
    static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() { return {}; }
};

} // namespace mlir
} // namespace boas

#endif // BOAS_MLIR_LIST_OPS_H
