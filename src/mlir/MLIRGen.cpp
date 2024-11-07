#include "boas/MLIRGen.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include <iostream>

namespace boas {

mlir::Value MLIRGenerator::generateExpression(ExprNode* node) {
  if (auto numberNode = dynamic_cast<NumberNode*>(node)) {
    return generateNumber(numberNode);
  } else if (auto binaryNode = dynamic_cast<BinaryOpNode*>(node)) {
    return generateBinaryOp(binaryNode);
  }
  return nullptr;
}

mlir::Value MLIRGenerator::generateNumber(NumberNode* node) {
  auto type = builder.getF64Type();
  auto value = builder.getF64FloatAttr(node->getValue());
  return builder.create<mlir::arith::ConstantOp>(builder.getUnknownLoc(), type, value);
}

mlir::Value MLIRGenerator::generateBinaryOp(BinaryOpNode* node) {
  auto left = generateExpression(const_cast<ExprNode*>(node->getLeft()));
  auto right = generateExpression(const_cast<ExprNode*>(node->getRight()));
  
  if (!left || !right) return nullptr;
  
  switch (node->getOp()) {
    case BinaryOpNode::ADD:
      return builder.create<mlir::arith::AddFOp>(
        builder.getUnknownLoc(), left, right);
    case BinaryOpNode::SUB:
      return builder.create<mlir::arith::SubFOp>(
        builder.getUnknownLoc(), left, right);
    case BinaryOpNode::MUL:
      return builder.create<mlir::arith::MulFOp>(
        builder.getUnknownLoc(), left, right);
    case BinaryOpNode::DIV:
      return builder.create<mlir::arith::DivFOp>(
        builder.getUnknownLoc(), left, right);
  }
  return nullptr;
}

mlir::Operation* MLIRGenerator::generateModule(ASTNode* node) {
  auto moduleOp = mlir::ModuleOp::create(builder.getUnknownLoc());
  auto functionType = builder.getFunctionType({}, {});
  auto func = builder.create<mlir::func::FuncOp>(
    builder.getUnknownLoc(), "main", functionType);
  
  auto block = func.addEntryBlock();
  builder.setInsertionPointToStart(block);
  
  if (auto blockNode = dynamic_cast<BlockNode*>(node)) {
    for (const auto& stmt : blockNode->getStatements()) {
      if (auto printNode = dynamic_cast<PrintNode*>(stmt.get())) {
        if (auto exprNode = dynamic_cast<ExprNode*>(printNode->getExpression())) {
          auto value = generateExpression(exprNode);
          if (value) {
            builder.create<mlir::func::CallOp>(
              builder.getUnknownLoc(), 
              "printf",
              mlir::TypeRange{},
              mlir::ValueRange{value}
            );
          }
        }
      }
    }
  }
  
  builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
  moduleOp.push_back(func);
  return moduleOp.getOperation();
}

} // namespace boas