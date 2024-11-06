#include "boas/Lexer.h"

namespace boas {

void Lexer::skipWhitespace() {
  while (pos_ < input_.length() && isspace(input_[pos_])) {
    pos_++;
  }
}

Token Lexer::getNextToken() {
  skipWhitespace();

  if (pos_ >= input_.length()) {
    return {TOK_EOF, ""};
  }

  if (input_.substr(pos_, 5) == "print") {
    pos_ += 5;
    return {TOK_PRINT, "print"};
  }

  if (input_[pos_] == '(') {
    pos_++;
    return {TOK_LPAREN, "("};
  }

  if (input_[pos_] == ')') {
    pos_++;
    return {TOK_RPAREN, ")"};
  }

  if (input_[pos_] == '"') {
    std::string str;
    pos_++; // 跳过开始的引号
    while (pos_ < input_.length() && input_[pos_] != '"') {
      str += input_[pos_++];
    }
    pos_++; // 跳过结束的引号
    return {TOK_STRING, str};
  }

  return {TOK_EOF, ""};
}

} // namespace boas
