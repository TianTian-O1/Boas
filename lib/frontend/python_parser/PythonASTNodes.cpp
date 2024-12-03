#include "frontend/python_parser/PythonASTNodes.h"
#include "frontend/python_parser/PythonASTVisitor.h"

namespace boas {
namespace python {

std::unique_ptr<matrix::ExprAST> ModuleNode::accept(PythonASTVisitor* visitor) const {
    return visitor->visitModule(this);
}

std::unique_ptr<matrix::ExprAST> FunctionDefNode::accept(PythonASTVisitor* visitor) const {
    return visitor->visitFunctionDef(this);
}

std::unique_ptr<matrix::ExprAST> ExprNode::accept(PythonASTVisitor* visitor) const {
    return visitor->visitExpr(this);
}

std::unique_ptr<matrix::ExprAST> BinOpNode::accept(PythonASTVisitor* visitor) const {
    return visitor->visitBinOp(this);
}

std::unique_ptr<matrix::ExprAST> NumNode::accept(PythonASTVisitor* visitor) const {
    return visitor->visitNum(this);
}

std::unique_ptr<matrix::ExprAST> NameNode::accept(PythonASTVisitor* visitor) const {
    return visitor->visitName(this);
}

std::unique_ptr<matrix::ExprAST> AssignNode::accept(PythonASTVisitor* visitor) const {
    return visitor->visitAssign(this);
}

std::unique_ptr<matrix::ExprAST> CallNode::accept(PythonASTVisitor* visitor) const {
    return visitor->visitCall(this);
}

std::unique_ptr<matrix::ExprAST> ListNode::accept(PythonASTVisitor* visitor) const {
    return visitor->visitList(this);
}

std::unique_ptr<matrix::ExprAST> ForNode::accept(PythonASTVisitor* visitor) const {
    return visitor->visitFor(this);
}

std::unique_ptr<matrix::ExprAST> ReturnNode::accept(PythonASTVisitor* visitor) const {
    return visitor->visitReturn(this);
}

std::unique_ptr<matrix::ExprAST> StringNode::accept(PythonASTVisitor* visitor) const {
    return visitor->visitString(this);
}

} // namespace python
} // namespace boas 