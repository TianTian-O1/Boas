#include "boas/Lexer.h"
#include "boas/Parser.h"
#include "boas/MLIRGen.h"
#include "boas/SemanticAnalyzer.h"
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

  // 语义分析
  boas::SemanticAnalyzer analyzer;
  analyzer.analyze(ast.get());
  if (analyzer.hasErrors()) {
    for (const auto& error : analyzer.getErrors()) {
      std::cerr << "Semantic error: " << error.getMessage() << std::endl;
    }
    return 1;
  }

  // MLIR生成
  mlir::MLIRContext context;
  boas::MLIRGenerator generator(context);
  generator.generateModule(ast.get());  // 不需要存储返回值
  
  // 执行
  if (auto blockNode = dynamic_cast<boas::BlockNode*>(ast.get())) {
    for (const auto& stmt : blockNode->getStatements()) {
      if (auto printNode = dynamic_cast<boas::PrintNode*>(stmt.get())) {
        if (printNode->isStringExpr()) {
          std::cout << printNode->getStringValue() << std::endl;
        } else {
          std::cout << printNode->getEvaluatedValue() << std::endl;
        }
      }
    }
  } else if (auto printNode = dynamic_cast<boas::PrintNode*>(ast.get())) {
    std::cout << printNode->getEvaluatedValue() << std::endl;
  }

  return 0;
}
