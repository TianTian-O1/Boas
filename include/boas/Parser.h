#pragma once
#include "boas/Lexer.h"
#include "boas/AST.h"
#include <memory>
#include <vector>

namespace boas {

class Parser {
public:
  Parser(Lexer &lexer) : lexer_(lexer) {}
  std::unique_ptr<ASTNode> parse();

private:
  Lexer &lexer_;
  Token currentToken_;
  void advance();
  std::unique_ptr<ASTNode> parsePrint();
  std::unique_ptr<ASTNode> parseFunctionDef();
  std::vector<std::unique_ptr<ASTNode>> parseBlock();
  std::unique_ptr<ExprNode> parseExpression();
  std::unique_ptr<ExprNode> parseTerm();
  std::unique_ptr<ExprNode> parseFactor();
};

} // namespace boas
