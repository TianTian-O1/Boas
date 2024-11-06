#include "boas/Parser.h"

namespace boas {

void Parser::advance() {
  currentToken_ = lexer_.getNextToken();
}

std::unique_ptr<ASTNode> Parser::parse() {
  advance();
  
  if (currentToken_.type == TOK_PRINT) {
    advance();
    if (currentToken_.type == TOK_LPAREN) {
      advance();
      if (currentToken_.type == TOK_STRING) {
        std::string value = currentToken_.value;
        advance();
        if (currentToken_.type == TOK_RPAREN) {
          return std::make_unique<PrintNode>(value);
        }
      }
    }
  }
  
  return nullptr;
}

} // namespace boas
