//===----------------------------------------------------------------------===//
// Boas Operations Implementation
//===----------------------------------------------------------------------===//

#include "mlirops/BoasDialect/BoasDialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::boas;

//===----------------------------------------------------------------------===//
// MatmulOp Implementation
//===----------------------------------------------------------------------===//

std::pair<int64_t, int64_t> MatmulOp::getResultShape() {
  auto lhsType = getLhs().getType().cast<TensorType>();
  auto rhsType = getRhs().getType().cast<TensorType>();
  
  ArrayRef<int64_t> lhsShape = lhsType.getShape();
  ArrayRef<int64_t> rhsShape = rhsType.getShape();
  
  // 对于矩阵乘法 A(M,K) * B(K,N) = C(M,N)
  if (lhsShape.size() >= 2 && rhsShape.size() >= 2) {
    int64_t M = lhsShape[lhsShape.size() - 2];
    int64_t N = rhsShape[rhsShape.size() - 1];
    return {M, N};
  }
  
  return {-1, -1}; // 动态形状
}

StringRef MatmulOp::getOptimizationStrategy() {
  if (auto npuOpt = getNpuOpt()) {
    return npuOpt->getStrategy().getValue();
  }
  return "default";
}

LogicalResult MatmulOp::verify() {
  auto lhsType = getLhs().getType().cast<TensorType>();
  auto rhsType = getRhs().getType().cast<TensorType>();
  auto resultType = getResult().getType().cast<TensorType>();
  
  // 检查形状兼容性
  ArrayRef<int64_t> lhsShape = lhsType.getShape();
  ArrayRef<int64_t> rhsShape = rhsType.getShape();
  ArrayRef<int64_t> resultShape = resultType.getShape();
  
  if (lhsShape.size() < 2 || rhsShape.size() < 2) {
    return emitOpError("matrix multiplication requires at least 2D tensors");
  }
  
  // 检查内部维度匹配
  int64_t lhsK = lhsShape[lhsShape.size() - 1];
  int64_t rhsK = rhsShape[rhsShape.size() - 2];
  
  if (lhsK != -1 && rhsK != -1 && lhsK != rhsK) {
    return emitOpError("incompatible shapes for matrix multiplication: ")
           << "lhs inner dimension " << lhsK 
           << " does not match rhs inner dimension " << rhsK;
  }
  
  // 检查元素类型匹配
  if (lhsType.getElementType() != rhsType.getElementType()) {
    return emitOpError("operand element types must match");
  }
  
  // 检查设备匹配
  if (lhsType.getDevice() != rhsType.getDevice()) {
    return emitOpError("operands must be on the same device");
  }
  
  return success();
}

//===----------------------------------------------------------------------===//
// TensorCreateOp Implementation  
//===----------------------------------------------------------------------===//

LogicalResult TensorCreateOp::verify() {
  // 验证维度是否为正数
  auto rows = getRows();
  auto cols = getCols();
  
  // 如果是常量，检查值是否为正
  if (auto rowsConst = rows.getDefiningOp<arith::ConstantIndexOp>()) {
    if (rowsConst.value() <= 0) {
      return emitOpError("rows must be positive");
    }
  }
  
  if (auto colsConst = cols.getDefiningOp<arith::ConstantIndexOp>()) {
    if (colsConst.value() <= 0) {
      return emitOpError("cols must be positive");
    }
  }
  
  return success();
}

//===----------------------------------------------------------------------===//
// TensorRandomOp Implementation
//===----------------------------------------------------------------------===//

LogicalResult TensorRandomOp::verify() {
  // 类似TensorCreateOp的验证
  auto rows = getRows();
  auto cols = getCols();
  
  if (auto rowsConst = rows.getDefiningOp<arith::ConstantIndexOp>()) {
    if (rowsConst.value() <= 0) {
      return emitOpError("rows must be positive");
    }
  }
  
  if (auto colsConst = cols.getDefiningOp<arith::ConstantIndexOp>()) {
    if (colsConst.value() <= 0) {
      return emitOpError("cols must be positive");
    }
  }
  
  return success();
}

//===----------------------------------------------------------------------===//
// NPUKernelOp Implementation
//===----------------------------------------------------------------------===//

LogicalResult NPUKernelOp::verify() {
  // 验证输入输出数量匹配
  if (getInputs().size() != getOutputs().size()) {
    return emitOpError("number of inputs must match number of outputs");
  }
  
  // 验证kernel名称不为空
  if (getKernelName().empty()) {
    return emitOpError("kernel name cannot be empty");
  }
  
  // 验证body region
  if (getBody().empty()) {
    return emitOpError("kernel body cannot be empty");
  }
  
  return success();
}

//===----------------------------------------------------------------------===//
// NPULaunchOp Implementation
//===----------------------------------------------------------------------===//

LogicalResult NPULaunchOp::verify() {
  // 验证网格和块配置是否为正数
  auto checkPositive = [&](Value val, StringRef name) -> LogicalResult {
    if (auto constOp = val.getDefiningOp<arith::ConstantIndexOp>()) {
      if (constOp.value() <= 0) {
        return emitOpError(name + " must be positive");
      }
    }
    return success();
  };
  
  if (failed(checkPositive(getGridX(), "gridX")) ||
      failed(checkPositive(getGridY(), "gridY")) ||
      failed(checkPositive(getGridZ(), "gridZ")) ||
      failed(checkPositive(getBlockX(), "blockX")) ||
      failed(checkPositive(getBlockY(), "blockY")) ||
      failed(checkPositive(getBlockZ(), "blockZ"))) {
    return failure();
  }
  
  return success();
}

//===----------------------------------------------------------------------===//
// ToDeviceOp Implementation
//===----------------------------------------------------------------------===//

LogicalResult ToDeviceOp::verify() {
  auto inputType = getInput().getType().cast<TensorType>();
  auto resultType = getResult().getType().cast<TensorType>();
  
  // 验证除设备外的类型属性都相同
  if (inputType.getShape() != resultType.getShape() ||
      inputType.getElementType() != resultType.getElementType()) {
    return emitOpError("input and result must have same shape and element type");
  }
  
  // 验证目标设备不同
  if (inputType.getDevice() == resultType.getDevice()) {
    return emitOpError("target device must be different from input device");
  }
  
  return success();
}

//===----------------------------------------------------------------------===//
// ShapeOp Implementation  
//===----------------------------------------------------------------------===//

LogicalResult ShapeOp::verify() {
  auto inputType = getInput().getType().cast<TensorType>();
  ArrayRef<int64_t> shape = inputType.getShape();
  
  // 对于矩阵操作，确保至少是2D
  if (shape.size() < 2) {
    return emitOpError("shape operation requires at least 2D tensor");
  }
  
  return success();
}

//===----------------------------------------------------------------------===//
// 操作的构建器方法
//===----------------------------------------------------------------------===//

void MatmulOp::build(OpBuilder &builder, OperationState &result,
                     Value lhs, Value rhs, 
                     std::optional<NPUOptimizationAttr> npuOpt) {
  auto lhsType = lhs.getType().cast<TensorType>();
  auto rhsType = rhs.getType().cast<TensorType>();
  
  // 推断结果类型
  ArrayRef<int64_t> lhsShape = lhsType.getShape();
  ArrayRef<int64_t> rhsShape = rhsType.getShape();
  
  SmallVector<int64_t> resultShape;
  if (lhsShape.size() >= 2 && rhsShape.size() >= 2) {
    // 对于矩阵乘法 A(M,K) * B(K,N) = C(M,N)
    int64_t M = lhsShape[lhsShape.size() - 2];
    int64_t N = rhsShape[rhsShape.size() - 1];
    resultShape = {M, N};
  }
  
  auto resultType = TensorType::get(builder.getContext(), resultShape,
                                   lhsType.getElementType(), lhsType.getDevice());
  
  result.addOperands({lhs, rhs});
  result.addTypes(resultType);
  
  if (npuOpt) {
    result.addAttribute("npu_opt", *npuOpt);
  }
}

void TensorCreateOp::build(OpBuilder &builder, OperationState &result,
                          Value rows, Value cols, Value values, StringRef device) {
  // 从values推断元素类型
  Type elementType;
  if (auto shapedType = values.getType().dyn_cast<ShapedType>()) {
    elementType = shapedType.getElementType();
  } else {
    elementType = values.getType();
  }
  
  auto resultType = TensorType::get(builder.getContext(), {-1, -1}, 
                                   elementType, device);
  
  result.addOperands({rows, cols, values});
  result.addTypes(resultType);
  result.addAttribute("device", builder.getStringAttr(device));
}

void TensorRandomOp::build(OpBuilder &builder, OperationState &result,
                          Value rows, Value cols, StringRef device, 
                          Type elementType, std::optional<double> seed) {
  if (!elementType) {
    elementType = builder.getF32Type(); // 默认使用f32
  }
  
  auto resultType = TensorType::get(builder.getContext(), {-1, -1}, 
                                   elementType, device);
  
  result.addOperands({rows, cols});
  result.addTypes(resultType);
  result.addAttribute("device", builder.getStringAttr(device));
  
  if (seed) {
    result.addAttribute("seed", builder.getF64FloatAttr(*seed));
  }
}
