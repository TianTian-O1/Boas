#pragma once
#include <string>
#include <memory>
#include <vector>
#include "Value.h"

namespace boas {

// 基类
class ASTNode {
public:
  virtual ~ASTNode() = default;
};

// 表达式基类
class ExprNode : public ASTNode {
public:
  virtual ~ExprNode() = default;
};

// 将 StringNode 移到这里，在 PrintNode 之前
class StringNode : public ExprNode {
public:
  StringNode(const std::string& value) : value_(value) {}
  const std::string& getValue() const { return value_; }
private:
  std::string value_;
};

class NumberNode : public ExprNode {
public:
  NumberNode(double value) : value_(value) {}
  double getValue() const { return value_; }
private:
  double value_;
};

class BinaryOpNode : public ExprNode {
public:
  enum OpType { 
    ADD, SUB, MUL, DIV, MOD, EXP, FLOOR_DIV,
    BITAND, BITOR, BITXOR,
    LSHIFT, RSHIFT,
    LT, GT, LE, GE, EQ, NE,
    AND, OR  // 添加逻辑运算符
  };
  
  BinaryOpNode(OpType op, std::unique_ptr<ExprNode> left,
               std::unique_ptr<ExprNode> right)
    : op_(op), left_(std::move(left)), right_(std::move(right)) {}
  
  OpType getOp() const { return op_; }
  const ExprNode* getLeft() const { return left_.get(); }
  const ExprNode* getRight() const { return right_.get(); }

private:
  OpType op_;
  std::unique_ptr<ExprNode> left_;
  std::unique_ptr<ExprNode> right_;
};

// 打印节点
class PrintNode : public ASTNode {
public:
  PrintNode(std::unique_ptr<ExprNode> expr) 
    : expr_(std::move(expr)) {}
  
  ExprNode* getExpression() const { return expr_.get(); }
  void setValue(const Value& val) { value_ = val; }
  const Value& getValue() const { return value_; }
  
  // 添加新的方法
  bool isStringExpr() const {
    return dynamic_cast<StringNode*>(expr_.get()) != nullptr;
  }
  
  std::string getStringValue() const {
    if (auto strNode = dynamic_cast<StringNode*>(expr_.get())) {
      return strNode->getValue();
    }
    return "";
  }
  
  double getEvaluatedValue() const {
    if (auto numNode = dynamic_cast<NumberNode*>(expr_.get())) {
      return numNode->getValue();
    }
    return 0.0;
  }

private:
  std::unique_ptr<ExprNode> expr_;
  Value value_;
};

// 函数节点
class FunctionNode : public ASTNode {
public:
  FunctionNode(const std::string& name, std::vector<std::unique_ptr<ASTNode>> body)
    : name_(name), body_(std::move(body)) {}
  
  const std::string& getName() const { return name_; }
  const std::vector<std::unique_ptr<ASTNode>>& getBody() const { return body_; }

private:
  std::string name_;
  std::vector<std::unique_ptr<ASTNode>> body_;
};

// 在 PrintNode 类之后，FunctionNode 类之前添加
class BlockNode : public ASTNode {
public:
  BlockNode(std::vector<std::unique_ptr<ASTNode>> statements)
    : statements_(std::move(statements)) {}
  
  const std::vector<std::unique_ptr<ASTNode>>& getStatements() const { 
    return statements_; 
  }

private:
  std::vector<std::unique_ptr<ASTNode>> statements_;
};

class AssignmentNode : public ASTNode {
public:
  AssignmentNode(const std::string& name, std::unique_ptr<ExprNode> expr)
    : name_(name), expr_(std::move(expr)) {}
  
  const std::string& getName() const { return name_; }
  ExprNode* getExpression() const { return expr_.get(); }

private:
  std::string name_;
  std::unique_ptr<ExprNode> expr_;
};

class BooleanNode : public ExprNode {
public:
  BooleanNode(bool value) : value_(value) {}
  bool getValue() const { return value_; }
private:
  bool value_;
};

class VariableNode : public ExprNode {
public:
  VariableNode(const std::string& name) : name_(name) {}
  const std::string& getName() const { return name_; }
private:
  std::string name_;
};

class UnaryOpNode : public ExprNode {
public:
  enum OpType { NOT };
  
  UnaryOpNode(OpType op, std::unique_ptr<ExprNode> operand)
    : op_(op), operand_(std::move(operand)) {}
  
  OpType getOp() const { return op_; }
  const ExprNode* getOperand() const { return operand_.get(); }

private:
  OpType op_;
  std::unique_ptr<ExprNode> operand_;
};

} // namespace boas
