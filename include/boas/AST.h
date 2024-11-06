#pragma once
#include <string>
#include <memory>
#include <vector>

namespace boas {

class ASTNode {
public:
  virtual ~ASTNode() = default;
};

class PrintNode : public ASTNode {
public:
  PrintNode(const std::string &value) : value_(value) {}
  const std::string &getValue() const { return value_; }

private:
  std::string value_;
};

class FunctionNode : public ASTNode {
public:
  FunctionNode(const std::string &name, 
              std::vector<std::unique_ptr<ASTNode>> body)
    : name_(name), body_(std::move(body)) {}
  
  const std::string &getName() const { return name_; }
  const std::vector<std::unique_ptr<ASTNode>> &getBody() const { return body_; }

private:
  std::string name_;
  std::vector<std::unique_ptr<ASTNode>> body_;
};

class BlockNode : public ASTNode {
public:
  BlockNode(std::vector<std::unique_ptr<ASTNode>> stmts) 
    : statements(std::move(stmts)) {}
  
  const std::vector<std::unique_ptr<ASTNode>>& getStatements() const {
    return statements;
  }

private:
  std::vector<std::unique_ptr<ASTNode>> statements;
};

} // namespace boas
