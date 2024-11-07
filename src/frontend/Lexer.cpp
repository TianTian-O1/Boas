#include "boas/Lexer.h"
#include <iostream>
#include <ostream>

namespace boas {

const std::unordered_map<std::string, TokenType> keywords = {
  {"and",      TOK_AND},
  {"as",       TOK_AS},
  {"assert",   TOK_ASSERT},
  {"break",    TOK_BREAK},
  {"class",    TOK_CLASS},
  {"continue", TOK_CONTINUE},
  {"def",      TOK_DEF},
  {"del",      TOK_DEL},
  {"elif",     TOK_ELIF},
  {"else",     TOK_ELSE},
  {"except",   TOK_EXCEPT},
  {"False",    TOK_FALSE},
  {"finally",  TOK_FINALLY},
  {"for",      TOK_FOR},
  {"from",     TOK_FROM},
  {"global",   TOK_GLOBAL},
  {"if",       TOK_IF},
  {"import",   TOK_IMPORT},
  {"in",       TOK_IN},
  {"is",       TOK_IS},
  {"lambda",   TOK_LAMBDA},
  {"None",     TOK_NONE},
  {"nonlocal", TOK_NONLOCAL},
  {"not",      TOK_NOT},
  {"or",       TOK_OR},
  {"pass",     TOK_PASS},
  {"print",    TOK_PRINT},
  {"raise",    TOK_RAISE},
  {"return",   TOK_RETURN},
  {"True",     TOK_TRUE},
  {"try",      TOK_TRY},
  {"while",    TOK_WHILE},
  {"with",     TOK_WITH},
  {"yield",    TOK_YIELD}
};

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
    std::string num;
    bool hasDecimal = false;
    
    while (pos_ < input_.length() && 
           (isdigit(input_[pos_]) || (!hasDecimal && input_[pos_] == '.'))) {
      if (input_[pos_] == '.') {
        hasDecimal = true;
      }
      num += input_[pos_];
      pos_++;
    }
    return {TOK_NUMBER, num};
  }

  // 处理标识符和关键字
  if (isalpha(input_[pos_]) || input_[pos_] == '_') {
    std::string identifier;
    while (pos_ < input_.length() && 
           (isalnum(input_[pos_]) || input_[pos_] == '_')) {
      identifier += input_[pos_];
      pos_++;
    }
    
    if (identifier == "print") return {TOK_PRINT, identifier};
    if (identifier == "def") return {TOK_DEF, identifier};
    if (identifier == "true") return {TOK_TRUE, identifier};
    if (identifier == "false") return {TOK_FALSE, identifier};
    
    return {TOK_IDENTIFIER, identifier};
  }

  // 处理字符串
  if (input_[pos_] == '"' || input_[pos_] == '\'') {
    char quote = input_[pos_];
    pos_++;
    std::string str;
    while (pos_ < input_.length() && input_[pos_] != quote) {
      str += input_[pos_];
      pos_++;
    }
    if (pos_ < input_.length()) {
      pos_++;
      return {TOK_STRING, str};
    }
  }

  // 处理布尔值
  if (input_.substr(pos_, 4) == "true") {
    pos_ += 4;
    return {TOK_TRUE, "true"};
  }
  if (input_.substr(pos_, 5) == "false") {
    pos_ += 5;
    return {TOK_FALSE, "false"};
  }

  // 处理赋值操作符
  if (input_[pos_] == '=') {
    pos_++;
    return {TOK_ASSIGN, "="};
  }

  // 处理运算符
  switch (input_[pos_]) {
    case '+': pos_++; return {TOK_PLUS, "+"};
    case '-': pos_++; return {TOK_MINUS, "-"};
    case '*': pos_++; return {TOK_MUL, "*"};
    case '/': pos_++; return {TOK_DIV, "/"};
    case '=': pos_++; return {TOK_ASSIGN, "="};
    case '(': pos_++; return {TOK_LPAREN, "("};
    case ')': pos_++; return {TOK_RPAREN, ")"};
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

  return {TOK_EOF, ""};
}

} // namespace boas