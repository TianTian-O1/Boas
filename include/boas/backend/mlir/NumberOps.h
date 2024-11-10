#ifndef BOAS_NUMBER_OPS_H
#define BOAS_NUMBER_OPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

namespace boas {
namespace mlir {

class NumberConstantOp : public ::mlir::Op<NumberConstantOp,
                                          ::mlir::OpTrait::OneResult,
                                          ::mlir::OpTrait::ZeroSuccessors> {
public:
    using Op::Op;
    
    static void build(::mlir::OpBuilder &builder, ::mlir::OperationState &state,
                     int64_t value) {
        auto type = builder.getI64Type();
        auto attr = builder.getI64IntegerAttr(value);
        state.addAttribute("value", attr);
        state.addTypes(type);
    }
    
    int64_t getValue() {
        return (*this)->getAttrOfType<::mlir::IntegerAttr>("value").getInt();
    }
    
    ::mlir::Value getResult() { return this->getOperation()->getResult(0); }
    
    static ::llvm::StringRef getOperationName() { return "boas.constant"; }
    static ::mlir::LogicalResult verify() { return ::mlir::success(); }
    
    static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
        static ::llvm::StringRef names[] = {"value"};
        return ::llvm::ArrayRef<::llvm::StringRef>(names);
    }
};

} // namespace mlir
} // namespace boas

#endif // BOAS_NUMBER_OPS_H
