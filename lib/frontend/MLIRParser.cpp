#include "frontend/MLIRParser.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "frontend/MatrixDialect.h"

namespace matrix {

MLIRParser::MLIRParser(llvm::StringRef input) 
  : input_(input), 
    context_(std::make_unique<mlir::MLIRContext>()) {
  // 注册Matrix方言
  context_->loadDialect<MatrixDialect>();
}

mlir::OwningOpRef<mlir::ModuleOp> MLIRParser::parse() {
  // 创建解析状态
  mlir::ParserState state(input_, context_.get());
  
  // 设置解析规则
  state.addCustomDirective("import", [](mlir::OpBuilder &builder, 
                                      llvm::StringRef moduleName) {
    // 解析import语句
    auto loc = builder.getUnknownLoc();
    builder.create<ImportOp>(loc, moduleName);
    return mlir::success();
  });

  state.addCustomDirective("from_import", [](mlir::OpBuilder &builder,
                                           llvm::StringRef moduleName,
                                           llvm::StringRef funcName) {
    // 解析from import语句
    auto loc = builder.getUnknownLoc();
    builder.create<FromImportOp>(loc, moduleName, funcName);
    return mlir::success();
  });

  state.addCustomDirective("tensor", [](mlir::OpBuilder &builder,
                                      mlir::Attribute data) {
    // 解析tensor创建
    auto loc = builder.getUnknownLoc();
    builder.create<TensorCreateOp>(loc, data);
    return mlir::success();
  });

  state.addCustomDirective("matmul", [](mlir::OpBuilder &builder,
                                      mlir::Value lhs,
                                      mlir::Value rhs) {
    // 解析矩阵乘法
    auto loc = builder.getUnknownLoc();
    builder.create<MatMulOp>(loc, lhs, rhs);
    return mlir::success();
  });

  state.addCustomDirective("print", [](mlir::OpBuilder &builder,
                                     mlir::Value value) {
    // 解析打印语句
    auto loc = builder.getUnknownLoc();
    builder.create<PrintOp>(loc, value);
    return mlir::success();
  });

  // 执行解析
  auto moduleOp = mlir::parseSourceFile<mlir::ModuleOp>(state);
  if (!moduleOp)
    return nullptr;

  // 验证生成的MLIR模块
  if (failed(mlir::verify(*moduleOp))) {
    moduleOp->emitError("module verification failed");
    return nullptr;
  }

  return moduleOp;
}

} // namespace matrix 