#pragma once
#include "boas/AST.h"
#include "boas/Value.h"
#include <vector>
#include <string>
#include <memory>
#include <map>
#include <unordered_map>

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
  void analyzePrintNode(PrintNode* node);
  void analyzeAssignmentNode(AssignmentNode* node);
  void analyzeFunctionNode(FunctionNode* node);
  bool hasErrors() const { return !errors_.empty(); }
  const std::vector<SemanticError>& getErrors() const { return errors_; }

private:
  Value evaluateExpr(const ExprNode* expr);
  std::vector<SemanticError> errors_;
  std::unordered_map<std::string, Value> symbolTable_;
};

} // namespace boas