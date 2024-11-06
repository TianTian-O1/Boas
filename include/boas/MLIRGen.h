#pragma once
#include "boas/AST.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Operation.h"

namespace boas {

class MLIRGenerator {
public:
  MLIRGenerator(mlir::MLIRContext &context) 
    : builder(&context), context(context) {}
  
  mlir::Operation* generateModule(ASTNode *node) {
    // 暂时返回空，后续实现
    return nullptr;
  }

private:
  mlir::OpBuilder builder;
  mlir::MLIRContext &context;
};

} // namespace boas