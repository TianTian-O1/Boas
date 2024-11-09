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
    // 跳过空行和注释
    if (currentToken_.type == TOK_NEWLINE || currentToken_.type == TOK_COMMENT) {
      advance();
      continue;
    }
    
    // 解析语句
    std::unique_ptr<ASTNode> stmt = nullptr;
    if (currentToken_.type == TOK_PRINT) {
      stmt = parsePrint();
    } else if (currentToken_.type == TOK_IDENTIFIER) {
      stmt = parseAssignment();
    }
    
    if (stmt) {
      statements.push_back(std::move(stmt));
    }
    
    // 如果当前token不是换行符或EOF，说明有语法错误，跳过到下一行
    while (currentToken_.type != TOK_NEWLINE && currentToken_.type != TOK_EOF) {
      advance();
    }
    
    // 跳过换行符
    if (currentToken_.type == TOK_NEWLINE) {
      advance();
    }
  }
  
  return std::make_unique<BlockNode>(std::move(statements));
}

std::unique_ptr<ExprNode> Parser::parseFactor() {
  if (currentToken_.type == TOK_LPAREN) {
    advance();  // 跳过左括号
    auto expr = parseExpression();
    if (!expr || currentToken_.type != TOK_RPAREN) {
      return nullptr;
    }
    advance();  // 跳过右括号
    return expr;
  }
  
  if (currentToken_.type == TOK_NUMBER) {
    double value = std::stod(currentToken_.value);
    advance();
    return std::make_unique<NumberNode>(value);
  } else if (currentToken_.type == TOK_STRING) {
    std::string value = currentToken_.value;
    advance();
    return std::make_unique<StringNode>(value);
  } else if (currentToken_.type == TOK_TRUE) {
    advance();
    return std::make_unique<BooleanNode>(true);
  } else if (currentToken_.type == TOK_FALSE) {
    advance();
    return std::make_unique<BooleanNode>(false);
  } else if (currentToken_.type == TOK_IDENTIFIER) {
    std::string name = currentToken_.value;
    advance();
    return std::make_unique<VariableNode>(name);
  }
  return nullptr;
}

std::unique_ptr<ExprNode> Parser::parseTerm() {
  auto left = parseFactor();
  if (!left) return nullptr;
  
  while (currentToken_.type == TOK_MUL || 
         currentToken_.type == TOK_DIV || 
         currentToken_.type == TOK_MODULO) {
    Token op = currentToken_;
    advance();
    
    auto right = parseFactor();
    if (!right) return nullptr;
    
    BinaryOpNode::OpType opType;
    switch (op.type) {
      case TOK_MUL: opType = BinaryOpNode::MUL; break;
      case TOK_DIV: opType = BinaryOpNode::DIV; break;
      case TOK_MODULO: opType = BinaryOpNode::MOD; break;
      default: return nullptr;
    }
    
    left = std::make_unique<BinaryOpNode>(opType, std::move(left), std::move(right));
  }
  
  return left;
}

std::unique_ptr<ExprNode> Parser::parseExpression() {
  auto left = parseTerm();
  if (!left) {
    return nullptr;
  }
  
  while (currentToken_.type == TOK_PLUS || currentToken_.type == TOK_MINUS ||
         currentToken_.type == TOK_LT || currentToken_.type == TOK_GT ||
         currentToken_.type == TOK_LE || currentToken_.type == TOK_GE ||
         currentToken_.type == TOK_EQ || currentToken_.type == TOK_NE) {
    BinaryOpNode::OpType op;
    switch (currentToken_.type) {
      case TOK_PLUS: op = BinaryOpNode::ADD; break;
      case TOK_MINUS: op = BinaryOpNode::SUB; break;
      case TOK_LT: op = BinaryOpNode::LT; break;
      case TOK_GT: op = BinaryOpNode::GT; break;
      case TOK_LE: op = BinaryOpNode::LE; break;
      case TOK_GE: op = BinaryOpNode::GE; break;
      case TOK_EQ: op = BinaryOpNode::EQ; break;
      case TOK_NE: op = BinaryOpNode::NE; break;
      default: return nullptr;
    }
    advance();
    
    auto right = parseTerm();
    if (!right) {
      return nullptr;
    }
    
    left = std::make_unique<BinaryOpNode>(op, std::move(left), std::move(right));
  }
  
  return left;
}

std::unique_ptr<ASTNode> Parser::parseAssignment() {
  std::string name = currentToken_.value;
  advance();  // 跳过变量名
  
  if (currentToken_.type != TOK_ASSIGN) {
    return nullptr;
  }
  advance();  // 跳过 =
  
  auto expr = parseExpression();
  if (!expr) {
    return nullptr;
  }
  
  return std::make_unique<AssignmentNode>(name, std::move(expr));
}

} // namespace boas
