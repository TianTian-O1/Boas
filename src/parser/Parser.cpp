#include "boas/Parser.h"
#include <iostream>

namespace boas {

void Parser::advance() {
  currentToken_ = lexer_.getNextToken();
}

std::unique_ptr<ASTNode> Parser::parsePrint() {
  advance(); // 跳过print
  if (currentToken_.type == TOK_LPAREN) {
    advance();
    if (currentToken_.type == TOK_STRING) {
      std::string value = currentToken_.value;
      advance();
      if (currentToken_.type == TOK_RPAREN) {
        advance();
        return std::make_unique<PrintNode>(value);
      }
    }
  }
  return nullptr;
}

std::vector<std::unique_ptr<ASTNode>> Parser::parseBlock() {
  std::vector<std::unique_ptr<ASTNode>> statements;
  
  // 如果有缩进，处理缩进
  if (currentToken_.type == TOK_INDENT) {
    advance();
  }

  while (currentToken_.type != TOK_DEDENT && 
         currentToken_.type != TOK_EOF) {
    if (currentToken_.type == TOK_PRINT) {
      auto stmt = parsePrint();
      if (stmt) {
        statements.push_back(std::move(stmt));
      }
    }
    if (currentToken_.type == TOK_NEWLINE) {
      advance();
    }
  }
  
  if (currentToken_.type == TOK_DEDENT) {
    advance();
  }
  
  return statements;
}

std::unique_ptr<ASTNode> Parser::parseFunctionDef() {
  advance(); // 跳过def
  
  if (currentToken_.type != TOK_IDENTIFIER) {
    return nullptr;
  }
  std::string name = currentToken_.value;
  advance();
  
  if (currentToken_.type != TOK_LPAREN) {
    return nullptr;
  }
  advance();
  
  if (currentToken_.type != TOK_RPAREN) {
    return nullptr;
  }
  advance();
  
  if (currentToken_.type != TOK_COLON) {
    return nullptr;
  }
  advance();
  
  if (currentToken_.type != TOK_NEWLINE) {
    return nullptr;
  }
  advance();
  
  auto body = parseBlock();
  return std::make_unique<FunctionNode>(name, std::move(body));
}

std::unique_ptr<ASTNode> Parser::parse() {
  advance();
  
  if (currentToken_.type == TOK_DEF) {
    return parseFunctionDef();
  }
  
  if (currentToken_.type == TOK_PRINT) {
    return parsePrint();
  }
  
  // 处理顶层语句
  auto statements = parseBlock();
  if (!statements.empty()) {
    // 如果只有一个语句，直接返回
    if (statements.size() == 1) {
      return std::move(statements[0]);
    }
    // 否则创建一个Block节点
    return std::make_unique<BlockNode>(std::move(statements));
  }
  
  return nullptr;
}

} // namespace boas
