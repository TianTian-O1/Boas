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
    if (!expr) {
      return nullptr;
    }
    if (currentToken_.type != TOK_RPAREN) {
      return nullptr;
    }
    advance();
    return std::make_unique<PrintNode>(std::move(expr));
  }
  return nullptr;
}

std::vector<std::unique_ptr<ASTNode>> Parser::parseBlock() {
  std::vector<std::unique_ptr<ASTNode>> statements;
  
  if (currentToken_.type == TOK_INDENT) {
    advance();
  }
  
  while (currentToken_.type != TOK_DEDENT && currentToken_.type != TOK_EOF) {
    if (currentToken_.type == TOK_PRINT) {
      auto stmt = parsePrint();
      if (stmt) {
        statements.push_back(std::move(stmt));
      }
      while (currentToken_.type != TOK_NEWLINE && currentToken_.type != TOK_EOF) {
        advance();
      }
    } else if (currentToken_.type == TOK_IDENTIFIER) {
      auto stmt = parseAssignment();
      if (stmt) {
        statements.push_back(std::move(stmt));
      }
    } else if (currentToken_.type == TOK_NEWLINE) {
      advance();
      continue;
    } else {
      advance();
    }
  }
  
  return statements;
}

std::unique_ptr<ASTNode> Parser::parseFunctionDef() {
  advance(); // 跳过 def
  if (currentToken_.type != TOK_IDENTIFIER) {
    return nullptr;
  }
  
  std::string funcName = currentToken_.value;
  advance();
  
  if (currentToken_.type != TOK_LPAREN) {
    return nullptr;
  }
  advance();
  
  // 暂时跳过参数解析
  if (currentToken_.type != TOK_RPAREN) {
    return nullptr;
  }
  advance();
  
  if (currentToken_.type != TOK_COLON) {
    return nullptr;
  }
  advance();
  
  // 处理函数体
  if (currentToken_.type == TOK_NEWLINE) {
    advance();
  }
  
  auto body = parseBlock();
  return std::make_unique<FunctionNode>(funcName, std::move(body));
}

std::unique_ptr<ASTNode> Parser::parse() {
  std::vector<std::unique_ptr<ASTNode>> statements;
  advance();  // 获取第一个token
  
  while (currentToken_.type != TOK_EOF) {
    if (currentToken_.type == TOK_DEF) {
      auto stmt = parseFunctionDef();
      if (stmt) {
        statements.push_back(std::move(stmt));
      }
    } else if (currentToken_.type == TOK_PRINT) {
      auto stmt = parsePrint();
      if (stmt) {
        statements.push_back(std::move(stmt));
      }
    } else if (currentToken_.type == TOK_IDENTIFIER) {
      auto stmt = parseAssignment();
      if (stmt) {
        statements.push_back(std::move(stmt));
      }
    }
    
    // 处理换行或前进到下一个token
    if (currentToken_.type == TOK_NEWLINE) {
      advance();
    } else if (!statements.empty() && statements.back()) {
      advance();
    } else {
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
  
  if (currentToken_.type == TOK_STRING) {
    std::string value = currentToken_.value;
    advance();
    return std::make_unique<StringNode>(value);
  }
  
  if (currentToken_.type == TOK_TRUE || currentToken_.type == TOK_FALSE) {
    bool value = (currentToken_.type == TOK_TRUE);
    advance();
    return std::make_unique<BooleanNode>(value);
  }
  
  if (currentToken_.type == TOK_IDENTIFIER) {
    std::string name = currentToken_.value;
    advance();
    return std::make_unique<VariableNode>(name);
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
  
  while (currentToken_.type == TOK_MUL || currentToken_.type == TOK_DIV) {
    auto op = currentToken_.type == TOK_MUL ? BinaryOpNode::MUL : BinaryOpNode::DIV;
    advance();
    auto right = parseFactor();
    left = std::make_unique<BinaryOpNode>(op, std::move(left), std::move(right));
  }
  
  return left;
}

std::unique_ptr<ExprNode> Parser::parseExpression() {
  auto left = parseTerm();
  
  while (currentToken_.type == TOK_PLUS || currentToken_.type == TOK_MINUS) {
    auto op = currentToken_.type == TOK_PLUS ? BinaryOpNode::ADD : BinaryOpNode::SUB;
    advance();
    auto right = parseTerm();
    left = std::make_unique<BinaryOpNode>(op, std::move(left), std::move(right));
  }
  
  return left;
}

std::unique_ptr<ASTNode> Parser::parseAssignment() {
  std::string varName = currentToken_.value;
  advance();
  
  if (currentToken_.type != TOK_ASSIGN) {
    return nullptr;
  }
  advance();
  
  auto expr = parseExpression();
  if (!expr) {
    return nullptr;
  }
  
  return std::make_unique<AssignmentNode>(varName, std::move(expr));
}

} // namespace boas
