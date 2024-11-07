#include "boas/Lexer.h"

namespace boas {

void Lexer::skipWhitespace() {
  while (pos_ < input_.length() && 
         (input_[pos_] == ' ' || input_[pos_] == '\t')) {
    pos_++;
  }
}

int Lexer::getIndentLevel() {
  int spaces = 0;
  size_t curr_pos = pos_;
  
  // 跳过换行符后的空格
  while (curr_pos < input_.length() && 
         (input_[curr_pos] == ' ' || input_[curr_pos] == '\t')) {
    if (input_[curr_pos] == ' ') {
      spaces++;
    } else if (input_[curr_pos] == '\t') {
      spaces += 4; // 将tab转换为4个空格
    }
    curr_pos++;
  }
  
  return spaces / 4;
}

Token Lexer::getNextToken() {
  skipWhitespace();
  
  if (pos_ >= input_.length()) {
    return {TOK_EOF, ""};
  }

  // 处理数字
  if (isdigit(input_[pos_]) || input_[pos_] == '.') {
    std::string number;
    bool hasDecimal = false;
    
    while (pos_ < input_.length() && 
           (isdigit(input_[pos_]) || input_[pos_] == '.')) {
      if (input_[pos_] == '.') {
        if (hasDecimal) break;
        hasDecimal = true;
      }
      number += input_[pos_++];
    }
    return {TOK_NUMBER, number};
  }

  // 处理运算符
  switch (input_[pos_]) {
    case '+': pos_++; return {TOK_PLUS, "+"};
    case '-': pos_++; return {TOK_MINUS, "-"};
    case '*': pos_++; return {TOK_MULTIPLY, "*"};
    case '/': pos_++; return {TOK_DIVIDE, "/"};
  }

  // 处理换行和缩进
  if (input_[pos_] == '\n') {
    pos_++;
    skipWhitespace();
    int new_indent = getIndentLevel();
    
    if (new_indent > indent_level_) {
      indent_level_ = new_indent;
      return {TOK_INDENT, std::string(4 * new_indent, ' ')};
    } else if (new_indent < indent_level_) {
      indent_level_ = new_indent;
      return {TOK_DEDENT, ""};
    }
    return {TOK_NEWLINE, "\n"};
  }

  // 处理关键字和标识符
  if (isalpha(input_[pos_]) || input_[pos_] == '_') {
    std::string identifier;
    while (pos_ < input_.length() && 
           (isalnum(input_[pos_]) || input_[pos_] == '_')) {
      identifier += input_[pos_++];
    }
    
    if (identifier == "print") {
      return {TOK_PRINT, identifier};
    } else if (identifier == "def") {
      return {TOK_DEF, identifier};
    }
    return {TOK_IDENTIFIER, identifier};
  }

  // 处理标点符号
  if (input_[pos_] == '(') {
    pos_++;
    return {TOK_LPAREN, "("};
  }

  if (input_[pos_] == ')') {
    pos_++;
    return {TOK_RPAREN, ")"};
  }

  if (input_[pos_] == ':') {
    pos_++;
    return {TOK_COLON, ":"};
  }

  // 处理字符串
  if (input_[pos_] == '"') {
    pos_++; // 跳过开始的引号
    std::string str;
    while (pos_ < input_.length() && input_[pos_] != '"') {
      str += input_[pos_++];
    }
    if (pos_ < input_.length() && input_[pos_] == '"') {
      pos_++; // 跳过结束的引号
      return {TOK_STRING, str};
    }
  }

  return {TOK_EOF, ""};
}

} // namespace boas