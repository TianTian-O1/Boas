#pragma once
#include <string>
#include <unordered_map>

namespace boas {

enum TokenType {
  TOK_EOF,
  TOK_IDENTIFIER,
  TOK_NUMBER,
  TOK_STRING,
  
  // 关键字
  TOK_AND,
  TOK_AS,
  TOK_ASSERT,
  TOK_BREAK,
  TOK_CLASS,
  TOK_CONTINUE,
  TOK_DEF,
  TOK_DEL,
  TOK_ELIF,
  TOK_ELSE,
  TOK_EXCEPT,
  TOK_FALSE,
  TOK_FINALLY,
  TOK_FOR,
  TOK_FROM,
  TOK_GLOBAL,
  TOK_IF,
  TOK_IMPORT,
  TOK_IN,
  TOK_IS,
  TOK_LAMBDA,
  TOK_NONE,
  TOK_NONLOCAL,
  TOK_NOT,
  TOK_OR,
  TOK_PASS,
  TOK_PRINT,
  TOK_RAISE,
  TOK_RETURN,
  TOK_TRUE,
  TOK_TRY,
  TOK_WHILE,
  TOK_WITH,
  TOK_YIELD,
  
  // 运算符和分隔符
  TOK_PLUS,
  TOK_MINUS,
  TOK_MUL,
  TOK_DIV,
  TOK_MULTIPLY,
  TOK_DIVIDE,
  TOK_ASSIGN,
  TOK_LPAREN,
  TOK_RPAREN,
  TOK_COLON,
  TOK_INDENT,
  TOK_DEDENT,
  TOK_NEWLINE
};

struct Token {
  TokenType type;
  std::string value;
  
  // 添加默认构造函数
  Token() : type(TOK_EOF), value("") {}
  Token(TokenType t, const std::string &v) : type(t), value(v) {}
};

// 关键字映射表
extern const std::unordered_map<std::string, TokenType> keywords;

} // namespace boas