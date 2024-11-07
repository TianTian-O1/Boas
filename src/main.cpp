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
  if (!file.is_open()) {
    std::cerr << "Failed to open file: " << argv[1] << std::endl;
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
      std::cerr << "Semantic error: " << error.getMessage() << std::endl;
    }
    return 1;
  }

  mlir::MLIRContext context;
  boas::MLIRGenerator generator(context);
  generator.generateModule(ast.get());
  
  // 执行
  if (auto blockNode = dynamic_cast<boas::BlockNode*>(ast.get())) {
    for (const auto& stmt : blockNode->getStatements()) {
      if (auto printNode = dynamic_cast<boas::PrintNode*>(stmt.get())) {
        analyzer.analyzePrintNode(printNode);
        const auto& value = printNode->getValue();
        if (value.isString()) {
          std::cout << value.asString() << std::endl;
        } else if (value.isNumber()) {
          std::cout << value.asNumber() << std::endl;
        } else if (value.isBoolean()) {
          std::cout << (value.asBoolean() ? "true" : "false") << std::endl;
        }
      } else if (auto assignNode = dynamic_cast<boas::AssignmentNode*>(stmt.get())) {
        analyzer.analyzeAssignmentNode(assignNode);
      } else if (auto funcNode = dynamic_cast<boas::FunctionNode*>(stmt.get())) {
        for (const auto& funcStmt : funcNode->getBody()) {
          if (auto printNode = dynamic_cast<boas::PrintNode*>(funcStmt.get())) {
            analyzer.analyzePrintNode(printNode);
            const auto& value = printNode->getValue();
            if (value.isString()) {
              std::cout << value.asString() << std::endl;
            } else if (value.isNumber()) {
              std::cout << value.asNumber() << std::endl;
            } else if (value.isBoolean()) {
              std::cout << (value.asBoolean() ? "true" : "false") << std::endl;
            }
          }
        }
      }
    }
  }
  return 0;
}
