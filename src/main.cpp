#include "boas/Lexer.h"
#include "boas/Parser.h"
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
  
  boas::Lexer lexer(buffer.str());
  boas::Parser parser(lexer);
  
  auto ast = parser.parse();
  if (auto printNode = dynamic_cast<boas::PrintNode*>(ast.get())) {
    std::cout << printNode->getValue() << std::endl;
  }

  return 0;
}
