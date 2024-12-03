#ifndef BOAS_PYTHON_AST_NODES_H
#define BOAS_PYTHON_AST_NODES_H

#include <memory>
#include <string>
#include <vector>
#include "frontend/AST.h"

namespace boas {
namespace python {

class PythonASTVisitor;

// AST节点基类
class PythonASTNode {
public:
    virtual ~PythonASTNode() = default;
    virtual std::unique_ptr<matrix::ExprAST> accept(PythonASTVisitor* visitor) const = 0;
};

// 模块节点
class ModuleNode : public PythonASTNode {
    std::vector<std::unique_ptr<PythonASTNode>> body;
public:
    ModuleNode(std::vector<std::unique_ptr<PythonASTNode>> body)
        : body(std::move(body)) {}
    
    const std::vector<std::unique_ptr<PythonASTNode>>& getBody() const { return body; }
    
    std::unique_ptr<matrix::ExprAST> accept(PythonASTVisitor* visitor) const override;
};

// 函数定义节点
class FunctionDefNode : public PythonASTNode {
    std::string name;
    std::vector<std::string> args;
    std::vector<std::unique_ptr<PythonASTNode>> body;
public:
    FunctionDefNode(std::string name, std::vector<std::string> args,
                   std::vector<std::unique_ptr<PythonASTNode>> body)
        : name(std::move(name)), args(std::move(args)), body(std::move(body)) {}
    
    const std::string& getName() const { return name; }
    const std::vector<std::string>& getArgs() const { return args; }
    const std::vector<std::unique_ptr<PythonASTNode>>& getBody() const { return body; }
    
    std::unique_ptr<matrix::ExprAST> accept(PythonASTVisitor* visitor) const override;
};

// 表达式节点
class ExprNode : public PythonASTNode {
    std::unique_ptr<PythonASTNode> value;
public:
    ExprNode(std::unique_ptr<PythonASTNode> value)
        : value(std::move(value)) {}
    
    const PythonASTNode* getValue() const { return value.get(); }
    
    std::unique_ptr<matrix::ExprAST> accept(PythonASTVisitor* visitor) const override;
};

// 二元操作节点
class BinOpNode : public PythonASTNode {
    std::unique_ptr<PythonASTNode> left;
    std::unique_ptr<PythonASTNode> right;
    std::string op;
public:
    BinOpNode(std::unique_ptr<PythonASTNode> left,
              std::unique_ptr<PythonASTNode> right,
              std::string op)
        : left(std::move(left)), right(std::move(right)), op(std::move(op)) {}
    
    const PythonASTNode* getLeft() const { return left.get(); }
    const PythonASTNode* getRight() const { return right.get(); }
    const std::string& getOp() const { return op; }
    
    std::unique_ptr<matrix::ExprAST> accept(PythonASTVisitor* visitor) const override;
};

// 数字节点
class NumNode : public PythonASTNode {
    double value;
public:
    NumNode(double value) : value(value) {}
    
    double getValue() const { return value; }
    
    std::unique_ptr<matrix::ExprAST> accept(PythonASTVisitor* visitor) const override;
};

// 名称节点
class NameNode : public PythonASTNode {
    std::string id;
public:
    NameNode(std::string id) : id(std::move(id)) {}
    
    const std::string& getId() const { return id; }
    
    std::unique_ptr<matrix::ExprAST> accept(PythonASTVisitor* visitor) const override;
};

// 赋值节点
class AssignNode : public PythonASTNode {
    std::string target;
    std::unique_ptr<PythonASTNode> value;
public:
    AssignNode(std::string target, std::unique_ptr<PythonASTNode> value)
        : target(std::move(target)), value(std::move(value)) {}
    
    const std::string& getTarget() const { return target; }
    const PythonASTNode* getValue() const { return value.get(); }
    
    std::unique_ptr<matrix::ExprAST> accept(PythonASTVisitor* visitor) const override;
};

// 函数调用节点
class CallNode : public PythonASTNode {
    std::string func;
    std::vector<std::unique_ptr<PythonASTNode>> args;
public:
    CallNode(std::string func, std::vector<std::unique_ptr<PythonASTNode>> args)
        : func(std::move(func)), args(std::move(args)) {}
    
    const std::string& getFunc() const { return func; }
    const std::vector<std::unique_ptr<PythonASTNode>>& getArgs() const { return args; }
    
    std::unique_ptr<matrix::ExprAST> accept(PythonASTVisitor* visitor) const override;
};

// 列表节点
class ListNode : public PythonASTNode {
    std::vector<std::unique_ptr<PythonASTNode>> elements;
public:
    ListNode(std::vector<std::unique_ptr<PythonASTNode>> elements)
        : elements(std::move(elements)) {}
    
    const std::vector<std::unique_ptr<PythonASTNode>>& getElements() const { return elements; }
    
    std::unique_ptr<matrix::ExprAST> accept(PythonASTVisitor* visitor) const override;
};

// For循环节点
class ForNode : public PythonASTNode {
    std::string target;
    std::unique_ptr<PythonASTNode> iter;
    std::vector<std::unique_ptr<PythonASTNode>> body;
public:
    ForNode(std::string target, std::unique_ptr<PythonASTNode> iter,
            std::vector<std::unique_ptr<PythonASTNode>> body)
        : target(std::move(target)), iter(std::move(iter)), body(std::move(body)) {}
    
    const std::string& getTarget() const { return target; }
    const PythonASTNode* getIter() const { return iter.get(); }
    const std::vector<std::unique_ptr<PythonASTNode>>& getBody() const { return body; }
    
    std::unique_ptr<matrix::ExprAST> accept(PythonASTVisitor* visitor) const override;
};

// Return节点
class ReturnNode : public PythonASTNode {
    std::unique_ptr<PythonASTNode> value;
public:
    ReturnNode(std::unique_ptr<PythonASTNode> value)
        : value(std::move(value)) {}
    
    const PythonASTNode* getValue() const { return value.get(); }
    
    std::unique_ptr<matrix::ExprAST> accept(PythonASTVisitor* visitor) const override;
};

// 字符串节点
class StringNode : public PythonASTNode {
    std::string value;
public:
    StringNode(std::string value) : value(std::move(value)) {}
    
    const std::string& getValue() const { return value; }
    
    std::unique_ptr<matrix::ExprAST> accept(PythonASTVisitor* visitor) const override;
};

} // namespace python
} // namespace boas

#endif // BOAS_PYTHON_AST_NODES_H 