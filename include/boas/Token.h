#pragma once
#include <string>

namespace boas {

enum TokenType {
  TOK_EOF,
  TOK_PRINT,
  TOK_DEF,
  TOK_IDENTIFIER,
  TOK_STRING,
  TOK_NUMBER,
  TOK_PLUS,
  TOK_MINUS,
  TOK_MULTIPLY,
  TOK_DIVIDE,
  TOK_LPAREN,
  TOK_RPAREN,
  TOK_COLON,
  TOK_NEWLINE,
  TOK_INDENT,
  TOK_DEDENT
};

struct Token {
  TokenType type;
  std::string value;
  
  Token(TokenType t, const std::string &v) : type(t), value(v) {}
};

} // namespace boas