#include "boas/backend/mlir/TensorOps.h"

namespace boas {
namespace mlir {

void TensorCreateOp::build(::mlir::OpBuilder &builder, ::mlir::OperationState &state,
                          ::mlir::Type elementType, ArrayRef<int64_t> shape) {
    // 创建tensor类型
    auto tensorType = TensorType::get(elementType, shape, builder.getContext());
    state.addTypes(tensorType);
    
    // 添加shape属性
    state.addAttribute("shape", 
        builder.getI64ArrayAttr(llvm::ArrayRef<int64_t>(shape.begin(), shape.end())));
}

void TensorMatMulOp::build(::mlir::OpBuilder &builder, ::mlir::OperationState &state,
                          ::mlir::Value lhs, ::mlir::Value rhs) {
    // 获取输入tensor的类型
    auto lhsType = lhs.getType().cast<TensorType>();
    auto rhsType = rhs.getType().cast<TensorType>();
    
    // 验证矩阵维度是否匹配
    auto lhsShape = lhsType.getShape();
    auto rhsShape = rhsType.getShape();
    assert(lhsShape.size() == 2 && rhsShape.size() == 2 && 
           "MatMul requires 2D tensors");
    assert(lhsShape[1] == rhsShape[0] && 
           "Matrix dimensions must match for multiplication");
    
    // 计算结果tensor的shape
    std::vector<int64_t> resultShape{lhsShape[0], rhsShape[1]};
    
    // 创建结果tensor类型
    auto resultType = TensorType::get(lhsType.getElementType(), resultShape, 
                                    builder.getContext());
    
    // 添加操作数和结果类型
    state.addOperands({lhs, rhs});
    state.addTypes(resultType);
}

// 添加verify方法来验证操作的正确性
::mlir::LogicalResult TensorMatMulOp::verify() {
    auto lhsType = getOperand(0).getType().cast<TensorType>();
    auto rhsType = getOperand(1).getType().cast<TensorType>();
    
    if (lhsType.getShape().size() != 2 || rhsType.getShape().size() != 2)
        return emitOpError("requires 2D tensors");
        
    if (lhsType.getShape()[1] != rhsType.getShape()[0])
        return emitOpError("matrix dimensions must match");
        
    return ::mlir::success();
}

} // namespace mlir
} // namespace boas
