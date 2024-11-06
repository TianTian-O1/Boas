#pragma once
#include <string>
#include <vector>

namespace boas {

enum TokenType {
  TOK_EOF = 0,
  TOK_PRINT,
  TOK_STRING,
  TOK_LPAREN,
  TOK_RPAREN
};

struct Token {
  TokenType type;
  std::string value;
};

class Lexer {
public:
  Lexer(const std::string &input) : input_(input), pos_(0) {}
  Token getNextToken();

private:
  std::string input_;
  size_t pos_;
  void skipWhitespace();
};

} // namespace boas
