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
    virtual void dump(int indent = 0) const = 0;
};

// 模块节点
class ModuleNode : public PythonASTNode {
public:
    std::vector<std::unique_ptr<PythonASTNode>> body;
    
    ModuleNode(std::vector<std::unique_ptr<PythonASTNode>> body)
        : body(std::move(body)) {}
    
    std::unique_ptr<matrix::ExprAST> accept(PythonASTVisitor* visitor) const override;
    void dump(int indent = 0) const override;
};

// 函数定义节点
class FunctionDefNode : public PythonASTNode {
public:
    std::string name;
    std::vector<std::string> args;
    std::vector<std::unique_ptr<PythonASTNode>> body;
    
    FunctionDefNode(std::string name, std::vector<std::string> args,
                   std::vector<std::unique_ptr<PythonASTNode>> body)
        : name(std::move(name)), args(std::move(args)), body(std::move(body)) {}
    
    std::unique_ptr<matrix::ExprAST> accept(PythonASTVisitor* visitor) const override;
    void dump(int indent = 0) const override;
};

// 表达式节点
class ExprNode : public PythonASTNode {
public:
    std::unique_ptr<PythonASTNode> value;
    
    ExprNode(std::unique_ptr<PythonASTNode> value)
        : value(std::move(value)) {}
    
    std::unique_ptr<matrix::ExprAST> accept(PythonASTVisitor* visitor) const override;
    void dump(int indent = 0) const override;
};

// 二元操作节点
class BinOpNode : public PythonASTNode {
public:
    std::unique_ptr<PythonASTNode> left;
    std::unique_ptr<PythonASTNode> right;
    std::string op;
    
    BinOpNode(std::unique_ptr<PythonASTNode> left,
              std::unique_ptr<PythonASTNode> right,
              std::string op)
        : left(std::move(left)), right(std::move(right)), op(std::move(op)) {}
    
    std::unique_ptr<matrix::ExprAST> accept(PythonASTVisitor* visitor) const override;
    void dump(int indent = 0) const override;
};

// 数字节点
class NumNode : public PythonASTNode {
public:
    double value;
    
    NumNode(double value) : value(value) {}
    
    std::unique_ptr<matrix::ExprAST> accept(PythonASTVisitor* visitor) const override;
    void dump(int indent = 0) const override;
};

// 名称节点
class NameNode : public PythonASTNode {
public:
    std::string id;
    
    NameNode(std::string id) : id(std::move(id)) {}
    
    std::unique_ptr<matrix::ExprAST> accept(PythonASTVisitor* visitor) const override;
    void dump(int indent = 0) const override;
};

// 赋值节点
class AssignNode : public PythonASTNode {
public:
    std::string target;
    std::unique_ptr<PythonASTNode> value;
    
    AssignNode(std::string target, std::unique_ptr<PythonASTNode> value)
        : target(std::move(target)), value(std::move(value)) {}
    
    std::unique_ptr<matrix::ExprAST> accept(PythonASTVisitor* visitor) const override;
    void dump(int indent = 0) const override;
};

// 函数调用节点
class CallNode : public PythonASTNode {
public:
    std::string callee;
    std::vector<std::unique_ptr<PythonASTNode>> arguments;
    
    CallNode(std::string callee, std::vector<std::unique_ptr<PythonASTNode>> arguments)
        : callee(std::move(callee)), arguments(std::move(arguments)) {}
    
    std::unique_ptr<matrix::ExprAST> accept(PythonASTVisitor* visitor) const override;
    void dump(int indent = 0) const override;
};

// 列表节点
class ListNode : public PythonASTNode {
public:
    std::vector<std::unique_ptr<PythonASTNode>> elements;
    
    ListNode(std::vector<std::unique_ptr<PythonASTNode>> elements)
        : elements(std::move(elements)) {}
    
    std::unique_ptr<matrix::ExprAST> accept(PythonASTVisitor* visitor) const override;
    void dump(int indent = 0) const override;
};

// For循环节点
class ForNode : public PythonASTNode {
public:
    std::string target;
    std::unique_ptr<PythonASTNode> iter;
    std::vector<std::unique_ptr<PythonASTNode>> body;
    
    ForNode(std::string target,
           std::unique_ptr<PythonASTNode> iter,
           std::vector<std::unique_ptr<PythonASTNode>> body)
        : target(std::move(target)), iter(std::move(iter)), body(std::move(body)) {}
    
    std::unique_ptr<matrix::ExprAST> accept(PythonASTVisitor* visitor) const override;
    void dump(int indent = 0) const override;
};

// Return节点
class ReturnNode : public PythonASTNode {
public:
    std::unique_ptr<PythonASTNode> value;
    
    ReturnNode(std::unique_ptr<PythonASTNode> value)
        : value(std::move(value)) {}
    
    std::unique_ptr<matrix::ExprAST> accept(PythonASTVisitor* visitor) const override;
    void dump(int indent = 0) const override;
};

// 字符串节点
class StringNode : public PythonASTNode {
public:
    std::string value;
    
    StringNode(std::string value) : value(std::move(value)) {}
    
    std::unique_ptr<matrix::ExprAST> accept(PythonASTVisitor* visitor) const override;
    void dump(int indent = 0) const override;
};

// 属性访问节点
class AttributeNode : public PythonASTNode {
public:
    std::unique_ptr<PythonASTNode> value;
    std::string attr;
    
    AttributeNode(std::unique_ptr<PythonASTNode> value, std::string attr)
        : value(std::move(value)), attr(std::move(attr)) {}
    
    std::unique_ptr<matrix::ExprAST> accept(PythonASTVisitor* visitor) const override;
    void dump(int indent = 0) const override;
};

// 方法调用节点
class MethodCallNode : public PythonASTNode {
public:
    std::unique_ptr<PythonASTNode> object;
    std::string method;
    std::vector<std::unique_ptr<PythonASTNode>> arguments;
    
    MethodCallNode(std::unique_ptr<PythonASTNode> object, std::string method,
                  std::vector<std::unique_ptr<PythonASTNode>> arguments)
        : object(std::move(object)), method(std::move(method)), arguments(std::move(arguments)) {}
    
    std::unique_ptr<matrix::ExprAST> accept(PythonASTVisitor* visitor) const override;
    void dump(int indent = 0) const override;
};

// 导入节点
class ImportNode : public PythonASTNode {
public:
    std::string name;
    
    ImportNode(std::string name) : name(std::move(name)) {}
    
    std::unique_ptr<matrix::ExprAST> accept(PythonASTVisitor* visitor) const override;
    void dump(int indent = 0) const override;
};

} // namespace python
} // namespace boas

#endif // BOAS_PYTHON_AST_NODES_H 