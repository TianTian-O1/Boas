//===- BoasOps.cpp - Boas dialect ops --------------------------*- C++ -*-===//
//
// Part of the Boas-NPU Project
//
//===----------------------------------------------------------------------===//

#include "Boas/Dialect/Boas/IR/BoasOps.h"
#include "Boas/Dialect/Boas/IR/BoasDialect.h"
#include "Boas/Dialect/Boas/IR/BoasTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionImplementation.h"

using namespace mlir;
using namespace mlir::boas;

//===----------------------------------------------------------------------===//
// TensorCreateOp
//===----------------------------------------------------------------------===//

LogicalResult TensorCreateOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location,
    ValueRange operands, DictionaryAttr attributes, OpaqueProperties properties,
    RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  // 从attributes中获取shape和element type
  // 这里简化处理，实际应该从value attribute推断
  return success();
}

//===----------------------------------------------------------------------===//
// MatMulOp
//===----------------------------------------------------------------------===//

LogicalResult MatMulOp::verify() {
  auto lhsType = getLhs().getType().cast<TensorType>();
  auto rhsType = getRhs().getType().cast<TensorType>();
  auto resultType = getResult().getType().cast<TensorType>();

  // 验证矩阵乘法的shape兼容性
  // LHS: [M, K], RHS: [K, N], Result: [M, N]
  auto lhsShape = lhsType.getShape();
  auto rhsShape = rhsType.getShape();
  auto resultShape = resultType.getShape();

  if (lhsShape.size() != 2 || rhsShape.size() != 2 || resultShape.size() != 2) {
    return emitOpError("matmul requires 2D tensors");
  }

  // 检查维度匹配
  if (lhsShape[1] != rhsShape[0]) {
    return emitOpError("incompatible dimensions for matmul: ")
           << lhsShape[1] << " vs " << rhsShape[0];
  }

  if (resultShape[0] != lhsShape[0] || resultShape[1] != rhsShape[1]) {
    return emitOpError("result shape mismatch");
  }

  return success();
}

LogicalResult MatMulOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location,
    ValueRange operands, DictionaryAttr attributes, OpaqueProperties properties,
    RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {

  auto lhsType = operands[0].getType().cast<TensorType>();
  auto rhsType = operands[1].getType().cast<TensorType>();

  auto lhsShape = lhsType.getShape();
  auto rhsShape = rhsType.getShape();

  // Result shape: [M, N]
  SmallVector<int64_t, 2> resultShape = {lhsShape[0], rhsShape[1]};

  auto resultType = TensorType::get(
      resultShape, lhsType.getElementType(), context);

  inferredReturnTypes.push_back(resultType);
  return success();
}

//===----------------------------------------------------------------------===//
// FuncOp (temporarily commented out)
//===----------------------------------------------------------------------===//

/*
ParseResult FuncOp::parse(OpAsmParser &parser, OperationState &result) {
  auto buildFuncType =
      [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
         function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void FuncOp::print(OpAsmPrinter &p) {
  function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}
*/

//===----------------------------------------------------------------------===//
// TableGen generated logic
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Boas/Dialect/Boas/IR/BoasOps.cpp.inc"
