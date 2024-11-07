#pragma once
#include "boas/AST.h"
#include <vector>
#include <string>
#include <memory>

namespace boas {

class SemanticError {
public:
  SemanticError(const std::string& msg) : message_(msg) {}
  const std::string& getMessage() const { return message_; }
private:
  std::string message_;
};

class SemanticAnalyzer {
public:
  void analyze(ASTNode* node);
  const std::vector<SemanticError>& getErrors() const { return errors_; }
  bool hasErrors() const { return !errors_.empty(); }

private:
  void analyzePrintNode(PrintNode* node);
  double evaluateExpression(const ExprNode* expr);
  void analyzeBinaryOp(BinaryOpNode* node);
  
  std::vector<SemanticError> errors_;
};

} // namespace boas