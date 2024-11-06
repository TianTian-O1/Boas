#pragma once
#include <string>
#include <memory>

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

} // namespace boas
