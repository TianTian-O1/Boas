#pragma once
#include "boas/Token.h"
#include <string>

namespace boas {

class Lexer {
public:
  Lexer(const std::string& input) : input_(input), pos_(0), indent_level_(0) {}
  Token getNextToken();

private:
  void skipWhitespace();
  int getIndentLevel();
  void advance() { pos_++; }
  
  std::string input_;
  size_t pos_;
  int indent_level_;
};

} // namespace boas
