#ifndef BOAS_MLIR_PRINT_OPS_H
#define BOAS_MLIR_PRINT_OPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace boas {
namespace mlir {

class PrintOp : public ::mlir::Op<PrintOp,
                                 ::mlir::OpTrait::OneResult,
                                 ::mlir::OpTrait::ZeroSuccessors> {
public:
    using Op::Op;
    
    static void build(::mlir::OpBuilder &builder, ::mlir::OperationState &state,
                     ::mlir::Value value) {
        state.addOperands(value);
        state.addTypes(value.getType());
    }
    
    static ::llvm::StringRef getOperationName() { return "boas.print"; }
    static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() { return {}; }
    
    ::mlir::Value getResult() { return this->getOperation()->getResult(0); }
    ::mlir::Value getOperand() { return this->getOperation()->getOperand(0); }
    
    ::mlir::LogicalResult verify() {
        return ::mlir::success();
    }
    
    ::mlir::LogicalResult print() {
        llvm::outs() << "\n[DEBUG] PrintOp::print() called\n";
        
        // 获取操作数
        auto operand = getOperand();
        auto definingOp = operand.getDefiningOp();
        
        llvm::outs() << "[DEBUG] Operand type: " << operand.getType() << "\n";
        llvm::outs() << "[DEBUG] DefiningOp: " << (definingOp ? definingOp->getName().getStringRef().str() : "null") << "\n";
        
        // 处理常量
        if (auto constOp = ::llvm::dyn_cast_or_null<NumberConstantOp>(definingOp)) {
            llvm::outs() << "[DEBUG] Found NumberConstantOp\n";
            llvm::outs() << constOp.getValue() << "\n";
            return ::mlir::success();
        }
        
        // 处理列表
        if (auto listOp = ::llvm::dyn_cast_or_null<ListCreateOp>(definingOp)) {
            llvm::outs() << "[DEBUG] Found ListCreateOp\n";
            llvm::outs() << "[";
            
            // 收集所有元素
            std::vector<int64_t> elements;
            for (auto& use : listOp.getResult().getUses()) {
                llvm::outs() << "[DEBUG] Processing use: " << use.getOwner()->getName().getStringRef().str() << "\n";
                
                if (auto appendOp = ::llvm::dyn_cast<ListAppendOp>(use.getOwner())) {
                    llvm::outs() << "[DEBUG] Found ListAppendOp\n";
                    
                    if (auto constOp = ::llvm::dyn_cast_or_null<NumberConstantOp>(
                        appendOp->getOperand(1).getDefiningOp())) {
                        llvm::outs() << "[DEBUG] Found appended constant: " << constOp.getValue() << "\n";
                        elements.push_back(constOp.getValue());
                    }
                }
            }
            
            // 打印元素
            for (size_t i = 0; i < elements.size(); ++i) {
                if (i > 0) llvm::outs() << ", ";
                llvm::outs() << elements[i];
            }
            
            llvm::outs() << "]\n";
            return ::mlir::success();
        }
        
        llvm::outs() << "[DEBUG] No matching operation type found\n";
        return ::mlir::success();
    }
    
    ::mlir::LogicalResult printList(ListCreateOp listOp) {
        llvm::outs() << "[";
        
        bool first = true;
        for (auto operand : listOp.getOperation()->getOperands()) {
            if (!first) llvm::outs() << ", ";
            first = false;
            
            if (auto nestedListOp = llvm::dyn_cast_or_null<ListCreateOp>(
                operand.getDefiningOp())) {
                // 递归打印嵌套列表
                printList(nestedListOp);
            } else if (auto constOp = llvm::dyn_cast_or_null<NumberConstantOp>(
                operand.getDefiningOp())) {
                llvm::outs() << constOp.getValue();
            }
        }
        
        llvm::outs() << "]";
        return ::mlir::success();
    }
};

} // namespace mlir
} // namespace boas

#endif // BOAS_MLIR_PRINT_OPS_H
