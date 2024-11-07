#include "boas/SemanticAnalyzer.h"
#include <iostream>

namespace boas {

void SemanticAnalyzer::analyze(ASTNode* node) {
  if (auto blockNode = dynamic_cast<BlockNode*>(node)) {
    for (const auto& stmt : blockNode->getStatements()) {
      if (auto printNode = dynamic_cast<PrintNode*>(stmt.get())) {
        analyzePrintNode(printNode);
      } else if (auto assignNode = dynamic_cast<AssignmentNode*>(stmt.get())) {
        analyzeAssignmentNode(assignNode);
      } else if (auto funcNode = dynamic_cast<FunctionNode*>(stmt.get())) {
        analyzeFunctionNode(funcNode);
      }
    }
  } else if (auto funcNode = dynamic_cast<FunctionNode*>(node)) {
    analyzeFunctionNode(funcNode);
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
    std::cout << "Variable not found: " << varNode->getName() << std::endl;
    return Value(0.0);
  }
  
  if (auto binNode = dynamic_cast<const BinaryOpNode*>(expr)) {
    Value left = evaluateExpr(binNode->getLeft());
    Value right = evaluateExpr(binNode->getRight());
    
    if (left.isNumber() && right.isNumber()) {
      double lval = left.asNumber();
      double rval = right.asNumber();
      
      switch (binNode->getOp()) {
        case BinaryOpNode::ADD: return Value(lval + rval);
        case BinaryOpNode::SUB: return Value(lval - rval);
        case BinaryOpNode::MUL: return Value(lval * rval);
        case BinaryOpNode::DIV: 
          if (rval != 0.0) {
            return Value(lval / rval);
          }
          errors_.emplace_back("Division by zero");
          return Value(0.0);
      }
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