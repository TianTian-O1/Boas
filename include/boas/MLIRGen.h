#pragma once
#include "boas/AST.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
class Operation;
class Value;
}

namespace boas {

class MLIRGenerator {
public:
  MLIRGenerator(mlir::MLIRContext &context) 
    : builder(&context), context(context) {
    context.loadDialect<mlir::func::FuncDialect>();
    context.loadDialect<mlir::arith::ArithDialect>();
  }
  
  mlir::Operation* generateModule(ASTNode *node);

private:
  mlir::OpBuilder builder;
  mlir::MLIRContext &context;
  
  mlir::Value generateExpression(ExprNode* node);
  mlir::Value generateBinaryOp(BinaryOpNode* node);
  mlir::Value generateNumber(NumberNode* node);
};

} // namespace boas