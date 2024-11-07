#pragma once
#include <string>
#include <memory>
#include <vector>

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
  enum OpType { ADD, SUB, MUL, DIV };
  
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
    : expr_(std::move(expr)), evaluatedValue_(0.0) {}
  
  ExprNode* getExpression() const { return expr_.get(); }
  
  void setEvaluatedValue(double value) { evaluatedValue_ = value; }
  double getEvaluatedValue() const { return evaluatedValue_; }
  
  bool isStringExpr() const { 
    return dynamic_cast<const StringNode*>(expr_.get()) != nullptr; 
  }
  
  std::string getStringValue() const {
    if (auto strNode = dynamic_cast<const StringNode*>(expr_.get())) {
      return strNode->getValue();
    }
    return "";
  }

private:
  std::unique_ptr<ExprNode> expr_;
  double evaluatedValue_;
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

} // namespace boas
