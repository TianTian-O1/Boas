#pragma once
#include <string>
#include <unordered_map>

namespace boas {

enum TokenType {
  TOK_EOF,
  TOK_NEWLINE,
  TOK_COMMENT,
  TOK_SEMICOLON,
  TOK_LPAREN,
  TOK_RPAREN,
  TOK_LBRACE,
  TOK_RBRACE,
  TOK_COLON,
  TOK_COMMA,
  TOK_DOT,
  TOK_PLUS,
  TOK_MINUS,
  TOK_STAR,
  TOK_SLASH,
  TOK_PERCENT,
  TOK_POWER,
  TOK_FLOOR_DIV,
  TOK_BITAND,
  TOK_BITOR,
  TOK_BITXOR,
  TOK_LSHIFT,
  TOK_RSHIFT,
  TOK_LT,
  TOK_GT,
  TOK_LE,
  TOK_GE,
  TOK_EQ,
  TOK_NE,
  TOK_AND,
  TOK_OR,
  TOK_NOT,
  TOK_ASSIGN,
  TOK_DEF,
  TOK_PRINT,
  TOK_IDENTIFIER,
  TOK_NUMBER,
  TOK_STRING,
  TOK_TRUE,
  TOK_FALSE,
  TOK_AS,
  TOK_ASSERT,
  TOK_BREAK,
  TOK_CLASS,
  TOK_CONTINUE,
  TOK_DEL,
  TOK_ELIF,
  TOK_ELSE,
  TOK_EXCEPT,
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
  TOK_INDENT,
  TOK_DEDENT,
  TOK_MUL,
  TOK_DIV,
  TOK_MODULO,
  TOK_NONLOCAL,
  TOK_PASS,
  TOK_RAISE,
  TOK_RETURN,
  TOK_TRY,
  TOK_WHILE,
  TOK_WITH,
  TOK_YIELD
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