#include "frontend/python_parser/PythonASTVisitor.h"
#include "frontend/python_parser/PythonASTNodes.h"
#include "frontend/ASTImpl.h"
#include <memory>
#include <iostream>

namespace boas {
namespace python {

std::unique_ptr<matrix::ExprAST> BoasASTConverter::visitBinOp(const BinOpNode* node) {
    auto left = node->left->accept(this);
    auto right = node->right->accept(this);
    
    if (node->op == "@") {
        return std::make_unique<matrix::MatmulExprASTImpl>(
            std::move(left),
            std::move(right)
        );
    }
    
    return std::make_unique<matrix::BinaryExprASTImpl>(
        std::move(left),
        std::move(right),
        node->op
    );
}

std::unique_ptr<matrix::ExprAST> BoasASTConverter::visitNum(const NumNode* node) {
    return std::make_unique<matrix::NumberExprASTImpl>(node->value);
}

std::unique_ptr<matrix::ExprAST> BoasASTConverter::visitName(const NameNode* node) {
    return std::make_unique<matrix::VariableExprASTImpl>(node->id);
}

std::unique_ptr<matrix::ExprAST> BoasASTConverter::visitAssign(const AssignNode* node) {
    auto value = node->value->accept(this);
    return std::make_unique<matrix::AssignmentExprASTImpl>(
        node->target,
        std::move(value)
    );
}

std::unique_ptr<matrix::ExprAST> BoasASTConverter::visitExpr(const ExprNode* node) {
    auto value = node->value->accept(this);
    return value;
}

std::unique_ptr<matrix::ExprAST> BoasASTConverter::visitList(const ListNode* node) {
    std::vector<std::unique_ptr<matrix::ExprAST>> elements;
    for (const auto& element : node->elements) {
        if (element) {
            elements.push_back(element->accept(this));
        }
    }
    return std::make_unique<matrix::ArrayExprASTImpl>(std::move(elements));
}

std::unique_ptr<matrix::ExprAST> BoasASTConverter::visitCall(const CallNode* node) {
    const std::string& funcName = node->callee;
    
    std::vector<std::unique_ptr<matrix::ExprAST>> args;
    for (const auto& arg : node->arguments) {
        if (arg) {
            args.push_back(arg->accept(this));
        }
    }
    
    if (funcName == "tensor.random") {
        if (args.size() != 2) {
            std::cerr << "Error: tensor.random requires exactly 2 arguments" << std::endl;
            return nullptr;
        }
        return std::make_unique<matrix::TensorRandomExprASTImpl>(
            std::move(args[0]),
            std::move(args[1])
        );
    } else if (funcName == "tensor.create") {
        if (args.size() != 3) {
            std::cerr << "Error: tensor.create requires exactly 3 arguments" << std::endl;
            return nullptr;
        }
        return std::make_unique<matrix::TensorCreateExprASTImpl>(
            std::move(args[0]),
            std::move(args[1]),
            std::move(args[2])
        );
    } else if (funcName == "tensor.matmul") {
        if (args.size() != 2) {
            std::cerr << "Error: tensor.matmul requires exactly 2 arguments" << std::endl;
            return nullptr;
        }
        return std::make_unique<matrix::MatmulExprASTImpl>(
            std::move(args[0]),
            std::move(args[1])
        );
    } else if (funcName == "print") {
        if (args.empty()) {
            return std::make_unique<matrix::PrintExprASTImpl>(
                std::make_unique<matrix::StringExprASTImpl>("")
            );
        }
        return std::make_unique<matrix::PrintExprASTImpl>(std::move(args[0]));
    } else if (funcName == "time.now") {
        return std::make_unique<matrix::TimeCallExprAST>();
    }
    
    return std::make_unique<matrix::CallExprASTImpl>(funcName, std::move(args));
}

std::unique_ptr<matrix::ExprAST> BoasASTConverter::visitModule(const ModuleNode* node) {
    std::vector<std::unique_ptr<matrix::ExprAST>> statements;
    for (const auto& stmt : node->body) {
        if (stmt) {
            statements.push_back(stmt->accept(this));
        }
    }
    return std::make_unique<matrix::ModuleASTImpl>(std::move(statements));
}

std::unique_ptr<matrix::ExprAST> BoasASTConverter::visitFunctionDef(const FunctionDefNode* node) {
    std::vector<std::unique_ptr<matrix::ExprAST>> body;
    for (const auto& stmt : node->body) {
        if (stmt) {
            body.push_back(stmt->accept(this));
        }
    }
    
    return std::make_unique<matrix::FunctionASTImpl>(
        node->name,
        node->args,
        std::move(body)
    );
}

std::unique_ptr<matrix::ExprAST> BoasASTConverter::visitFor(const ForNode* node) {
    auto target = std::make_unique<matrix::VariableExprASTImpl>(node->target);
    auto iter = node->iter->accept(this);
    
    std::vector<std::unique_ptr<matrix::ExprAST>> body;
    
    for (const auto& stmt : node->body) {
        if (stmt) {
            body.push_back(stmt->accept(this));
        }
    }
    
    return std::make_unique<matrix::ForExprASTImpl>(
        std::move(target),
        std::move(iter),
        std::move(body)
    );
}

std::unique_ptr<matrix::ExprAST> BoasASTConverter::visitReturn(const ReturnNode* node) {
    auto value = node->value->accept(this);
    return std::make_unique<matrix::ReturnASTImpl>(std::move(value));
}

std::unique_ptr<matrix::ExprAST> BoasASTConverter::visitString(const StringNode* node) {
    return std::make_unique<matrix::StringExprASTImpl>(node->value);
}

std::unique_ptr<matrix::ExprAST> BoasASTConverter::visitAttribute(const AttributeNode* node) {
    auto value = node->value->accept(this);
    const std::string& attr = node->attr;
    return std::make_unique<matrix::AttributeExprASTImpl>(std::move(value), attr);
}

std::unique_ptr<matrix::ExprAST> BoasASTConverter::visitMethodCall(const MethodCallNode* node) {
    auto value = node->object->accept(this);
    const std::string& method = node->method;
    
    std::vector<std::unique_ptr<matrix::ExprAST>> args;
    for (const auto& arg : node->arguments) {
        if (arg) {
            args.push_back(arg->accept(this));
        }
    }
    
    if (method == "to") {
        if (args.size() != 1) {
            std::cerr << "Error: to() method requires exactly 1 argument" << std::endl;
            return nullptr;
        }
        
        if (auto* stringExpr = dynamic_cast<matrix::StringExprASTImpl*>(args[0].get())) {
            return std::make_unique<matrix::DeviceTransferExprAST>(
                std::move(value),
                stringExpr->getValue()
            );
        } else {
            std::cerr << "Error: device argument must be a string" << std::endl;
            return nullptr;
        }
    }
    
    return std::make_unique<matrix::CallExprASTImpl>(
        method,
        std::move(args)
    );
}

std::unique_ptr<matrix::ExprAST> BoasASTConverter::visitImport(const ImportNode* node) {
    return std::make_unique<matrix::ImportASTImpl>(node->name);
}

} // namespace python
} // namespace boas