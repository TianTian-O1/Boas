#include "boas/LLVMGen.h"
#include <llvm/IR/Module.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/FileSystem.h>
#include <fstream>
#include <vector>

namespace boas {

LLVMGenerator::LLVMGenerator() {
  context_ = std::make_unique<llvm::LLVMContext>();
  module_ = std::make_unique<llvm::Module>("boas_module", *context_);
  
  // 直接设置目标平台和数据布局
  module_->setTargetTriple("x86_64-unknown-linux-gnu");
  module_->setDataLayout("e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128");
  builder_ = std::make_unique<llvm::IRBuilder<>>(*context_);
}

void LLVMGenerator::generateModule(ASTNode* node) {
  if (auto blockNode = dynamic_cast<BlockNode*>(node)) {
    // 创建main函数
    auto mainType = llvm::FunctionType::get(
      llvm::Type::getInt32Ty(*context_), 
      false
    );
    auto mainFunc = llvm::Function::Create(
      mainType,
      llvm::Function::ExternalLinkage,
      "main",
      module_.get()
    );
    
    // 创建入口基本块
    auto entry = llvm::BasicBlock::Create(*context_, "entry", mainFunc);
    builder_->SetInsertPoint(entry);
    
    // 生成语句
    for (const auto& stmt : blockNode->getStatements()) {
      if (auto printNode = dynamic_cast<PrintNode*>(stmt.get())) {
        generatePrint(printNode);
      } else if (auto funcNode = dynamic_cast<FunctionNode*>(stmt.get())) {
        // 生成函数定义
        auto funcType = llvm::FunctionType::get(
          llvm::Type::getVoidTy(*context_),
          false
        );
        auto func = llvm::Function::Create(
          funcType,
          llvm::Function::ExternalLinkage,
          funcNode->getName(),
          module_.get()
        );
        
        // 创建函数体基本块
        auto funcEntry = llvm::BasicBlock::Create(*context_, "entry", func);
        builder_->SetInsertPoint(funcEntry);
        
        // 生成函数体语句
        for (const auto& bodyStmt : funcNode->getBody()) {
          if (auto bodyPrintNode = dynamic_cast<PrintNode*>(bodyStmt.get())) {
            generatePrint(bodyPrintNode);
          }
        }
        
        // 函数返回
        builder_->CreateRetVoid();
        
        // 在main函数中调用该函数
        builder_->SetInsertPoint(entry);
        builder_->CreateCall(func);
      }
    }
    
    // 返回0
    builder_->CreateRet(llvm::ConstantInt::get(llvm::Type::getInt32Ty(*context_), 0));
  }
}

llvm::Value* LLVMGenerator::generatePrint(PrintNode* node) {
  // 获取printf函数类型
  std::vector<llvm::Type*> args;
  args.push_back(llvm::PointerType::get(llvm::Type::getInt8Ty(*context_), 0));
  
  auto printfType = llvm::FunctionType::get(
    llvm::Type::getInt32Ty(*context_),
    args,
    true
  );

  // 声明printf函数
  auto printfFunc = module_->getOrInsertFunction("printf", printfType);
  
  // 获取要打印的值
  const auto& value = node->getValue();
  
  // 根据值的类型创建不同的格式化字符串和值
  std::string format;
  llvm::Value* printValue;
  
  if (value.isString()) {
    format = "%s\n";
    auto strGlobal = builder_->CreateGlobalString(
      llvm::StringRef(value.asString()),
      "str"
    );
    printValue = builder_->CreateBitCast(
      strGlobal,
      llvm::PointerType::get(llvm::Type::getInt8Ty(*context_), 0)
    );
  } else if (value.isNumber()) {
    format = "%.2f\n";
    printValue = llvm::ConstantFP::get(
      llvm::Type::getDoubleTy(*context_),
      value.asNumber()
    );
  } else if (value.isBool()) {
    format = "%s\n";
    auto strGlobal = builder_->CreateGlobalString(
      llvm::StringRef(value.asBool() ? "true" : "false"),
      "bool"
    );
    printValue = builder_->CreateBitCast(
      strGlobal,
      llvm::PointerType::get(llvm::Type::getInt8Ty(*context_), 0)
    );
  }
  
  // 创建格式化字符串
  auto formatStr = builder_->CreateGlobalString(
    llvm::StringRef(format),
    "fmt"
  );
  auto formatPtr = builder_->CreateBitCast(
    formatStr,
    llvm::PointerType::get(llvm::Type::getInt8Ty(*context_), 0)
  );
  
  // 调用printf
  std::vector<llvm::Value*> args_values;
  args_values.push_back(formatPtr);
  args_values.push_back(printValue);
  return builder_->CreateCall(printfFunc, args_values);
}

void LLVMGenerator::writeIRToFile(const std::string& filename) {
  std::error_code EC;
  llvm::raw_fd_ostream dest(filename, EC);
  
  if (EC) {
    llvm::errs() << "Could not open file: " << EC.message();
    return;
  }
  
  module_->print(dest, nullptr);
  dest.flush();
}

} // namespace boas