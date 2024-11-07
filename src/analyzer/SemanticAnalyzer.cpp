#include "boas/SemanticAnalyzer.h"
#include <iostream>

namespace boas {

void SemanticAnalyzer::analyze(ASTNode* node) {
  if (auto blockNode = dynamic_cast<BlockNode*>(node)) {
    for (const auto& stmt : blockNode->getStatements()) {
      if (auto printNode = dynamic_cast<PrintNode*>(stmt.get())) {
        analyzePrintNode(printNode);
      }
    }
  }
}

void SemanticAnalyzer::analyzePrintNode(PrintNode* node) {
  if (!node->getExpression()) {
    errors_.emplace_back("Print statement requires an expression");
    return;
  }
  
  // 计算表达式的值并存储在PrintNode中
  double value = evaluateExpression(node->getExpression());
  node->setEvaluatedValue(value);
}

double SemanticAnalyzer::evaluateExpression(const ExprNode* expr) {
  if (const auto* numberNode = dynamic_cast<const NumberNode*>(expr)) {
    return numberNode->getValue();
  }
  if (dynamic_cast<const StringNode*>(expr)) {
    // 对于字符串，我们返回一个特殊值，比如0
    return 0.0;
  }
  
  if (const auto* binaryNode = dynamic_cast<const BinaryOpNode*>(expr)) {
    double left = evaluateExpression(binaryNode->getLeft());
    double right = evaluateExpression(binaryNode->getRight());
    
    switch (binaryNode->getOp()) {
      case BinaryOpNode::ADD: return left + right;
      case BinaryOpNode::SUB: return left - right;
      case BinaryOpNode::MUL: return left * right;
      case BinaryOpNode::DIV: 
        if (right == 0.0) {
          errors_.emplace_back("Division by zero");
          return 0.0;
        }
        return left / right;
    }
  }
  
  return 0.0;
}

} // namespace boas