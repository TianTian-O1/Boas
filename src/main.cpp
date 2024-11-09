#include "boas/Lexer.h"
#include "boas/Parser.h"
#include "boas/MLIRGen.h"
#include "boas/SemanticAnalyzer.h"
#include "boas/LLVMGen.h"
#include "mlir/IR/MLIRContext.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <map>

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "用法: " << argv[0] << " <run|build> <input_file> [-o output]" << std::endl;
    return 1;
  }

  std::string mode = argv[1];
  std::string inputFile = argv[2];
  std::string outputFile;

  // 处理-o选项
  for (int i = 3; i < argc; i++) {
    if (std::string(argv[i]) == "-o" && i + 1 < argc) {
      outputFile = argv[i + 1];
      i++;  // 跳过下一个参数
    }
  }

  if (mode != "run" && mode != "build") {
    std::cerr << "无效的模式: " << mode << std::endl;
    std::cerr << "用法: " << argv[0] << " <run|build> <input_file> [-o output]" << std::endl;
    return 1;
  }

  std::ifstream file(inputFile);
  if (!file.is_open()) {
    std::cerr << "无法打开文件: " << inputFile << std::endl;
    return 1;
  }

  std::stringstream buffer;
  buffer << file.rdbuf();
  
  boas::Lexer lexer(buffer.str());
  boas::Parser parser(lexer);
  auto ast = parser.parse();

  boas::SemanticAnalyzer analyzer;
  analyzer.analyze(ast.get());
  
  if (analyzer.hasErrors()) {
    for (const auto& error : analyzer.getErrors()) {
      std::cerr << "语义错误: " << error.getMessage() << std::endl;
    }
    return 1;
  }

  if (mode == "run") {
    if (auto blockNode = dynamic_cast<boas::BlockNode*>(ast.get())) {
      for (const auto& stmt : blockNode->getStatements()) {
        if (auto printNode = dynamic_cast<boas::PrintNode*>(stmt.get())) {
          const auto& value = printNode->getValue();
          if (value.isString()) {
            std::cout << value.asString() << std::endl;
          } else if (value.isNumber()) {
            std::cout << value.asNumber() << std::endl;
          } else if (value.isBool()) {
            std::cout << (value.asBool() ? "true" : "false") << std::endl;
          }
        }
      }
    }
  } else if (mode == "build") {
    // 生成可执行文件
    boas::LLVMGenerator llvmGen;
    llvmGen.generateModule(ast.get());
    
    // 如果没有指定输出文件名，则使用输入文件名（去掉扩展名）
    if (outputFile.empty()) {
      outputFile = inputFile;
      size_t ext = outputFile.find_last_of('.');
      if (ext != std::string::npos) {
        outputFile = outputFile.substr(0, ext);
      }
    }
    
    // 生成LLVM IR
    llvmGen.writeIRToFile(outputFile + ".ll");
    
    // 调用系统命令编译成可执行文件
    std::string cmd = "llc -relocation-model=static " + outputFile + ".ll -o " + outputFile + ".s && ";
    cmd += "gcc -no-pie " + outputFile + ".s -o " + outputFile;
    system(cmd.c_str());
    
    std::cout << "已生成可执行文件: " << outputFile << std::endl;
  }

  return 0;
}
