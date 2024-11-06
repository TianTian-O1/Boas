#pragma once
#include "boas/Lexer.h"
#include "boas/AST.h"
#include <memory>

namespace boas {

class Parser {
public:
  Parser(Lexer &lexer) : lexer_(lexer) {}
  std::unique_ptr<ASTNode> parse();

private:
  Lexer &lexer_;
  Token currentToken_;
  void advance();
};

} // namespace boas
