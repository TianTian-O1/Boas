#include "boas/Lexer.h"
#include "boas/Parser.h"
#include "boas/MLIRGen.h"
#include "mlir/IR/MLIRContext.h"
#include <iostream>
#include <fstream>
#include <sstream>

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <input_file>" << std::endl;
    return 1;
  }

  std::ifstream file(argv[1]);
  std::stringstream buffer;
  buffer << file.rdbuf();
  
  // 词法分析和语法分析
  boas::Lexer lexer(buffer.str());
  boas::Parser parser(lexer);
  auto ast = parser.parse();

  // MLIR生成
  mlir::MLIRContext context;
  
  boas::MLIRGenerator generator(context);
  auto *module = generator.generateModule(ast.get());
  
  // 输出MLIR
  if (module) {
    module->dump();
  }

  // 执行
  if (auto funcNode = dynamic_cast<boas::FunctionNode*>(ast.get())) {
    if (funcNode->getName() == "main") {
      for (const auto &stmt : funcNode->getBody()) {
        if (auto printNode = dynamic_cast<boas::PrintNode*>(stmt.get())) {
          std::cout << printNode->getValue() << std::endl;
        }
      }
    }
  } else if (auto printNode = dynamic_cast<boas::PrintNode*>(ast.get())) {
    // 直接处理顶层的print语句
    std::cout << printNode->getValue() << std::endl;
  }

  return 0;
}
