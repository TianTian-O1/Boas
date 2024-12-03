#ifndef BOAS_PYTHON_AST_VISITOR_H
#define BOAS_PYTHON_AST_VISITOR_H

#include "frontend/AST.h"
#include <memory>

namespace boas {
namespace python {

class ModuleNode;
class FunctionDefNode;
class ExprNode;
class BinOpNode;
class NumNode;
class NameNode;
class AssignNode;
class CallNode;
class ListNode;
class ForNode;
class ReturnNode;
class StringNode;

// 访问器基类
class PythonASTVisitor {
public:
    virtual ~PythonASTVisitor() = default;
    
    virtual std::unique_ptr<matrix::ExprAST> visitModule(const ModuleNode* node) = 0;
    virtual std::unique_ptr<matrix::ExprAST> visitFunctionDef(const FunctionDefNode* node) = 0;
    virtual std::unique_ptr<matrix::ExprAST> visitExpr(const ExprNode* node) = 0;
    virtual std::unique_ptr<matrix::ExprAST> visitBinOp(const BinOpNode* node) = 0;
    virtual std::unique_ptr<matrix::ExprAST> visitNum(const NumNode* node) = 0;
    virtual std::unique_ptr<matrix::ExprAST> visitName(const NameNode* node) = 0;
    virtual std::unique_ptr<matrix::ExprAST> visitAssign(const AssignNode* node) = 0;
    virtual std::unique_ptr<matrix::ExprAST> visitCall(const CallNode* node) = 0;
    virtual std::unique_ptr<matrix::ExprAST> visitList(const ListNode* node) = 0;
    virtual std::unique_ptr<matrix::ExprAST> visitFor(const ForNode* node) = 0;
    virtual std::unique_ptr<matrix::ExprAST> visitReturn(const ReturnNode* node) = 0;
    virtual std::unique_ptr<matrix::ExprAST> visitString(const StringNode* node) = 0;
};

// Boas AST转换器
class BoasASTConverter : public PythonASTVisitor {
public:
    std::unique_ptr<matrix::ExprAST> visitModule(const ModuleNode* node) override;
    std::unique_ptr<matrix::ExprAST> visitFunctionDef(const FunctionDefNode* node) override;
    std::unique_ptr<matrix::ExprAST> visitExpr(const ExprNode* node) override;
    std::unique_ptr<matrix::ExprAST> visitBinOp(const BinOpNode* node) override;
    std::unique_ptr<matrix::ExprAST> visitNum(const NumNode* node) override;
    std::unique_ptr<matrix::ExprAST> visitName(const NameNode* node) override;
    std::unique_ptr<matrix::ExprAST> visitAssign(const AssignNode* node) override;
    std::unique_ptr<matrix::ExprAST> visitCall(const CallNode* node) override;
    std::unique_ptr<matrix::ExprAST> visitList(const ListNode* node) override;
    std::unique_ptr<matrix::ExprAST> visitFor(const ForNode* node) override;
    std::unique_ptr<matrix::ExprAST> visitReturn(const ReturnNode* node) override;
    std::unique_ptr<matrix::ExprAST> visitString(const StringNode* node) override;
};

} // namespace python
} // namespace boas

#endif // BOAS_PYTHON_AST_VISITOR_H 