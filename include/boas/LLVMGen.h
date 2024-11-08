#pragma once
#include "boas/AST.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/raw_ostream.h"

namespace boas {

class LLVMGenerator {
public:
  LLVMGenerator();
  
  void generateModule(ASTNode* node);
  void dumpLLVM();
  void writeIRToFile(const std::string& filename);
  
private:
  std::unique_ptr<llvm::LLVMContext> context_;
  std::unique_ptr<llvm::Module> module_;
  std::unique_ptr<llvm::IRBuilder<>> builder_;
  
  llvm::Value* generatePrint(PrintNode* node);
  llvm::Value* generateExpr(ExprNode* node);
  void generateFunction(FunctionNode* node);
};

} // namespace boas