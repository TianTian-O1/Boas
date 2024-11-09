#include "boas/SemanticAnalyzer.h"
#include <iostream>

namespace boas {

void SemanticAnalyzer::analyze(ASTNode* node) {
  if (auto blockNode = dynamic_cast<BlockNode*>(node)) {
    for (const auto& stmt : blockNode->getStatements()) {
      if (auto assignNode = dynamic_cast<AssignmentNode*>(stmt.get())) {
        Value value = evaluateExpr(assignNode->getExpression());
        symbolTable_[assignNode->getName()] = value;
      } else if (auto printNode = dynamic_cast<PrintNode*>(stmt.get())) {
        Value value = evaluateExpr(printNode->getExpression());
        printNode->setValue(value);
      }
    }
  }
}

void SemanticAnalyzer::analyzePrintNode(PrintNode* node) {
  if (!node->getExpression()) {
    errors_.emplace_back("Print statement requires an expression");
    return;
  }
  node->setValue(evaluateExpr(node->getExpression()));
}

void SemanticAnalyzer::analyzeAssignmentNode(AssignmentNode* node) {
  if (!node->getExpression()) {
    errors_.emplace_back("Assignment requires an expression");
    return;
  }
  symbolTable_[node->getName()] = evaluateExpr(node->getExpression());
}

Value SemanticAnalyzer::evaluateExpr(const ExprNode* expr) {
  if (auto numNode = dynamic_cast<const NumberNode*>(expr)) {
    return Value(numNode->getValue());
  }
  
  if (auto strNode = dynamic_cast<const StringNode*>(expr)) {
    return Value(strNode->getValue());
  }
  
  if (auto boolNode = dynamic_cast<const BooleanNode*>(expr)) {
    return Value(boolNode->getValue());
  }
  
  if (auto varNode = dynamic_cast<const VariableNode*>(expr)) {
    auto it = symbolTable_.find(varNode->getName());
    if (it != symbolTable_.end()) {
      return it->second;
    }
    if (varNode->getName() == "true") {
      return Value(true);
    }
    if (varNode->getName() == "false") {
      return Value(false);
    }
    std::cout << "Variable not found: " << varNode->getName() << std::endl;
    return Value(0.0);
  }
  
  if (auto binNode = dynamic_cast<const BinaryOpNode*>(expr)) {
    Value left = evaluateExpr(binNode->getLeft());
    Value right = evaluateExpr(binNode->getRight());
    
    // 处理数值运算
    if (left.isNumber() && right.isNumber()) {
      double lval = left.asNumber();
      double rval = right.asNumber();
      
      switch (binNode->getOp()) {
        case BinaryOpNode::ADD: return Value(lval + rval);
        case BinaryOpNode::SUB: return Value(lval - rval);
        case BinaryOpNode::MUL: return Value(lval * rval);
        case BinaryOpNode::DIV: 
          if (rval != 0.0) return Value(lval / rval);
          errors_.emplace_back("Division by zero");
          return Value(0.0);
        case BinaryOpNode::MOD: return Value(std::fmod(lval, rval));
        case BinaryOpNode::EXP: return Value(std::pow(lval, rval));
        case BinaryOpNode::FLOOR_DIV: return Value(std::floor(lval / rval));
        case BinaryOpNode::BITAND: return Value(static_cast<int>(lval) & static_cast<int>(rval));
        case BinaryOpNode::BITOR: return Value(static_cast<int>(lval) | static_cast<int>(rval));
        case BinaryOpNode::BITXOR: return Value(static_cast<int>(lval) ^ static_cast<int>(rval));
        case BinaryOpNode::LSHIFT: return Value(static_cast<int>(lval) << static_cast<int>(rval));
        case BinaryOpNode::RSHIFT: return Value(static_cast<int>(lval) >> static_cast<int>(rval));
        case BinaryOpNode::LT: return Value(lval < rval);
        case BinaryOpNode::GT: return Value(lval > rval);
        case BinaryOpNode::LE: return Value(lval <= rval);
        case BinaryOpNode::GE: return Value(lval >= rval);
        case BinaryOpNode::EQ: return Value(lval == rval);
        case BinaryOpNode::NE: return Value(lval != rval);
        case BinaryOpNode::AND: return Value(lval && rval);
        case BinaryOpNode::OR: return Value(lval || rval);
      }
    }
    
    // 处理布尔运算
    if (left.isBool() && right.isBool()) {
      bool lval = left.asBool();
      bool rval = right.asBool();
      
      switch (binNode->getOp()) {
        case BinaryOpNode::AND: return Value(lval && rval);
        case BinaryOpNode::OR: return Value(lval || rval);
        case BinaryOpNode::EQ: return Value(lval == rval);
        case BinaryOpNode::NE: return Value(lval != rval);
        default: return Value(false);
      }
    }
  }
  
  // 处理一元运算符
  if (auto unaryNode = dynamic_cast<const UnaryOpNode*>(expr)) {
    Value operand = evaluateExpr(unaryNode->getOperand());
    if (operand.isBool() && unaryNode->getOp() == UnaryOpNode::NOT) {
      return Value(!operand.asBool());
    }
  }
  
  return Value(0.0);
}

void SemanticAnalyzer::analyzeFunctionNode(FunctionNode* node) {
  std::unordered_map<std::string, Value> localSymbols;
  auto oldSymbols = std::move(symbolTable_);
  symbolTable_ = std::move(localSymbols);
  
  for (const auto& stmt : node->getBody()) {
    if (auto printNode = dynamic_cast<PrintNode*>(stmt.get())) {
      analyzePrintNode(printNode);
    } else if (auto assignNode = dynamic_cast<AssignmentNode*>(stmt.get())) {
      analyzeAssignmentNode(assignNode);
    }
  }
  
  symbolTable_ = std::move(oldSymbols);
}

} // namespace boas