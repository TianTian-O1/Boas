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

std::unique_ptr<matrix::ExprAST> AttributeNode::accept(PythonASTVisitor* visitor) const {
    return visitor->visitAttribute(this);
}

std::unique_ptr<matrix::ExprAST> MethodCallNode::accept(PythonASTVisitor* visitor) const {
    return visitor->visitMethodCall(this);
}

std::unique_ptr<matrix::ExprAST> ImportNode::accept(PythonASTVisitor* visitor) const {
    return visitor->visitImport(this);
}

void ModuleNode::dump(int indent) const {
    std::cout << std::string(indent, ' ') << "module:" << std::endl;
    for (const auto& stmt : body) {
        if (stmt) {
            stmt->dump(indent + 2);
        } else {
            std::cout << std::string(indent + 2, ' ') << "<null statement>" << std::endl;
        }
    }
}

void AssignNode::dump(int indent) const {
    std::cout << std::string(indent, ' ') << "Assignment " << target << " = ";
    if (value) {
        value->dump(0);  // 直接打印值，不需要额外缩进
    } else {
        std::cout << "<null value>";
    }
    std::cout << std::endl;
}

void CallNode::dump(int indent) const {
    std::cout << std::string(indent, ' ') << "Call " << callee << "(";
    for (size_t i = 0; i < arguments.size(); ++i) {
        if (i > 0) std::cout << ", ";
        if (arguments[i]) {
            arguments[i]->dump(0);
        } else {
            std::cout << "<null arg>";
        }
    }
    std::cout << ")";
}

void MethodCallNode::dump(int indent) const {
    std::cout << std::string(indent, ' ');
    if (object) {
        object->dump(0);
    } else {
        std::cout << "<null object>";
    }
    std::cout << "." << method << "(";
    for (size_t i = 0; i < arguments.size(); ++i) {
        if (i > 0) std::cout << ", ";
        if (arguments[i]) {
            arguments[i]->dump(0);
        } else {
            std::cout << "<null arg>";
        }
    }
    std::cout << ")";
}

void AttributeNode::dump(int indent) const {
    std::cout << std::string(indent, ' ');
    if (value) {
        value->dump(0);
    } else {
        std::cout << "<null value>";
    }
    std::cout << "." << attr;
}

void NameNode::dump(int indent) const {
    std::cout << std::string(indent, ' ') << id;
}

void NumNode::dump(int indent) const {
    std::cout << std::string(indent, ' ') << value;
}

void StringNode::dump(int indent) const {
    std::cout << std::string(indent, ' ') << "\"" << value << "\"";
}

void ImportNode::dump(int indent) const {
    std::cout << std::string(indent, ' ') << "import " << name << std::endl;
}

void FunctionDefNode::dump(int indent) const {
    std::cout << std::string(indent, ' ') << "def " << name << "(";
    for (size_t i = 0; i < args.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << args[i];
    }
    std::cout << "):" << std::endl;
    
    for (const auto& stmt : body) {
        if (stmt) {
            stmt->dump(indent + 2);
        } else {
            std::cout << std::string(indent + 2, ' ') << "<null statement>" << std::endl;
        }
    }
}

void ReturnNode::dump(int indent) const {
    std::cout << std::string(indent, ' ') << "return ";
    if (value) {
        value->dump(0);
    } else {
        std::cout << "<null value>";
    }
    std::cout << std::endl;
}

void ListNode::dump(int indent) const {
    std::cout << std::string(indent, ' ') << "[";
    for (size_t i = 0; i < elements.size(); ++i) {
        if (i > 0) std::cout << ", ";
        if (elements[i]) {
            elements[i]->dump(0);
        } else {
            std::cout << "<null element>";
        }
    }
    std::cout << "]";
}

void ExprNode::dump(int indent) const {
    std::cout << std::string(indent, ' ');
    if (value) {
        value->dump(0);
    } else {
        std::cout << "<null expression>";
    }
    std::cout << std::endl;
}

void BinOpNode::dump(int indent) const {
    std::cout << std::string(indent, ' ') << "(";
    if (left) {
        left->dump(0);
    } else {
        std::cout << "<null left>";
    }
    std::cout << " " << op << " ";
    if (right) {
        right->dump(0);
    } else {
        std::cout << "<null right>";
    }
    std::cout << ")";
}

void ForNode::dump(int indent) const {
    std::cout << std::string(indent, ' ') << "for " << target << " in ";
    if (iter) {
        iter->dump(0);
    } else {
        std::cout << "<null iter>";
    }
    std::cout << ":" << std::endl;
    
    for (const auto& stmt : body) {
        if (stmt) {
            stmt->dump(indent + 2);
        } else {
            std::cout << std::string(indent + 2, ' ') << "<null statement>" << std::endl;
        }
    }
}

} // namespace python
} // namespace boas 