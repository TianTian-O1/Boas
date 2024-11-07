#pragma once
#include <string>
#include <vector>

namespace boas {

enum TokenType {
  TOK_EOF = 0,
  TOK_PRINT,
  TOK_STRING,
  TOK_NUMBER,
  TOK_PLUS,
  TOK_MINUS,
  TOK_MULTIPLY,
  TOK_DIVIDE,
  TOK_LPAREN,
  TOK_RPAREN,
  TOK_DEF,
  TOK_COLON,
  TOK_INDENT,
  TOK_DEDENT,
  TOK_NEWLINE,
  TOK_IDENTIFIER
};

struct Token {
  TokenType type;
  std::string value;
};

class Lexer {
public:
  Lexer(const std::string &input) : input_(input), pos_(0), indent_level_(0) {}
  Token getNextToken();

private:
  std::string input_;
  size_t pos_;
  int indent_level_;
  void skipWhitespace();
  int getIndentLevel();
};

} // namespace boas
