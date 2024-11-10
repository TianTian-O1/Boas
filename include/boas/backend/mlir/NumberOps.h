#ifndef BOAS_MLIR_NUMBER_OPS_H
#define BOAS_MLIR_NUMBER_OPS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace boas {
namespace mlir {

class NumberConstantOp : public ::mlir::Op<NumberConstantOp,
                                         ::mlir::OpTrait::ZeroOperands,
                                         ::mlir::OpTrait::OneResult,
                                         ::mlir::OpTrait::ZeroSuccessors> {
public:
    using Op::Op;
    
    static ::llvm::StringRef getOperationName() { return "boas.number.constant"; }
    
    static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() { 
        static ::llvm::StringRef attrNames[] = {"value"};
        return ::llvm::ArrayRef<::llvm::StringRef>(attrNames);
    }
    
    static void build(::mlir::OpBuilder &builder, ::mlir::OperationState &state,
                     double value) {
        state.addAttribute("value", builder.getF64FloatAttr(value));
        state.addTypes(builder.getF64Type());
    }
    
    double getValue() {
        return (*this)->getAttrOfType<::mlir::FloatAttr>("value").getValueAsDouble();
    }
    
    ::mlir::Value getResult() { return this->getOperation()->getResult(0); }
    
    ::mlir::LogicalResult verify() {
        if (!(*this)->hasAttr("value"))
            return emitOpError("requires 'value' attribute");
        return ::mlir::success();
    }
};

} // namespace mlir
} // namespace boas

#endif // BOAS_MLIR_NUMBER_OPS_H
