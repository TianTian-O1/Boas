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
    auto expr = parseExpression();
    if (expr && currentToken_.type == TOK_RPAREN) {
      advance();
      return std::make_unique<PrintNode>(std::move(expr));
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
  std::vector<std::unique_ptr<ASTNode>> statements;
  advance();  // 获取第一个token
  
  while (currentToken_.type != TOK_EOF) {
    if (currentToken_.type == TOK_PRINT) {
      auto stmt = parsePrint();
      if (stmt) {
        statements.push_back(std::move(stmt));
      }
    }
    if (currentToken_.type == TOK_NEWLINE) {
      advance();
    } else if (currentToken_.type != TOK_EOF) {
      advance();
    }
  }
  
  return std::make_unique<BlockNode>(std::move(statements));
}

std::unique_ptr<ExprNode> Parser::parseFactor() {
  if (currentToken_.type == TOK_NUMBER) {
    double value = std::stod(currentToken_.value);
    advance();
    return std::make_unique<NumberNode>(value);
  }
  
  if (currentToken_.type == TOK_LPAREN) {
    advance();
    auto expr = parseExpression();
    if (currentToken_.type == TOK_RPAREN) {
      advance();
      return expr;
    }
  }
  
  return nullptr;
}

std::unique_ptr<ExprNode> Parser::parseTerm() {
  auto left = parseFactor();
  
  while (currentToken_.type == TOK_MULTIPLY || currentToken_.type == TOK_DIVIDE) {
    BinaryOpNode::OpType op = (currentToken_.type == TOK_MULTIPLY) ? 
                              BinaryOpNode::MUL : BinaryOpNode::DIV;
    advance();
    auto right = parseFactor();
    left = std::make_unique<BinaryOpNode>(op, std::move(left), std::move(right));
  }
  
  return left;
}

std::unique_ptr<ExprNode> Parser::parseExpression() {
  if (currentToken_.type == TOK_STRING) {
    auto strNode = std::make_unique<StringNode>(currentToken_.value);
    advance();
    return strNode;
  }
  auto left = parseTerm();
  
  while (currentToken_.type == TOK_PLUS || currentToken_.type == TOK_MINUS) {
    BinaryOpNode::OpType op = (currentToken_.type == TOK_PLUS) ? 
                              BinaryOpNode::ADD : BinaryOpNode::SUB;
    advance();
    auto right = parseTerm();
    left = std::make_unique<BinaryOpNode>(op, std::move(left), std::move(right));
  }
  
  return left;
}

} // namespace boas
