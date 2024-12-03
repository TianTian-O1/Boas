#include "frontend/python_parser/PythonASTVisitor.h"
#include "frontend/python_parser/PythonASTNodes.h"
#include "frontend/ASTImpl.h"
#include <memory>
#include <string>

namespace boas {
namespace python {

std::unique_ptr<matrix::ExprAST> BoasASTConverter::visitBinOp(const BinOpNode* node) {
    auto left = node->getLeft()->accept(this);
    auto right = node->getRight()->accept(this);
    
    if (node->getOp() == "@") {
        return std::make_unique<matrix::MatmulExprASTImpl>(
            std::move(left),
            std::move(right)
        );
    } else {
        return std::make_unique<matrix::BinaryExprASTImpl>(
            std::move(left),
            std::move(right),
            node->getOp()
        );
    }
}

std::unique_ptr<matrix::ExprAST> BoasASTConverter::visitNum(const NumNode* node) {
    return std::make_unique<matrix::NumberExprASTImpl>(node->getValue());
}

std::unique_ptr<matrix::ExprAST> BoasASTConverter::visitName(const NameNode* node) {
    return std::make_unique<matrix::VariableExprASTImpl>(node->getId());
}

std::unique_ptr<matrix::ExprAST> BoasASTConverter::visitAssign(const AssignNode* node) {
    auto value = node->getValue()->accept(this);
    return std::make_unique<matrix::AssignmentExprASTImpl>(
        node->getTarget(),
        std::move(value)
    );
}

std::unique_ptr<matrix::ExprAST> BoasASTConverter::visitExpr(const ExprNode* node) {
    auto value = node->getValue()->accept(this);
    
    // 如果表达式的值是一个 PrintExprASTImpl，直接返回它
    if (auto* printExpr = dynamic_cast<matrix::PrintExprASTImpl*>(value.get())) {
        return value;
    }
    
    // 如果是 print 函数调用，包装它
    if (auto* callExpr = dynamic_cast<matrix::CallExprAST*>(value.get())) {
        if (callExpr->getCallee() == "print") {
            if (!callExpr->getArgs().empty()) {
                return std::make_unique<matrix::PrintExprASTImpl>(
                    std::unique_ptr<matrix::ExprAST>(callExpr->getArgs()[0]->clone())
                );
            }
            return std::make_unique<matrix::PrintExprASTImpl>(
                std::make_unique<matrix::StringExprASTImpl>("")
            );
        }
    }
    
    // 直接返回表达式的值
    return value;
}

std::unique_ptr<matrix::ExprAST> BoasASTConverter::visitList(const ListNode* node) {
    std::vector<std::unique_ptr<matrix::ExprAST>> elements;
    for (const auto& element : node->getElements()) {
        if (auto expr = element->accept(this)) {
            elements.push_back(std::move(expr));
        }
    }
    return std::make_unique<matrix::ArrayExprASTImpl>(std::move(elements));
}

std::unique_ptr<matrix::ExprAST> BoasASTConverter::visitCall(const CallNode* node) {
    const std::string& funcName = node->getFunc();
    std::vector<std::unique_ptr<matrix::ExprAST>> args;
    
    for (const auto& arg : node->getArgs()) {
        args.push_back(arg->accept(this));
    }
    
    // 检查是否是属性访问（例如 tensor.create）
    size_t dotPos = funcName.find('.');
    if (dotPos != std::string::npos) {
        std::string objName = funcName.substr(0, dotPos);
        std::string methodName = funcName.substr(dotPos + 1);
        
        // 处理 tensor 相关的方法调用
        if (objName == "tensor") {
            if (methodName == "matmul" && args.size() == 2) {
                return std::make_unique<matrix::MatmulExprASTImpl>(
                    std::move(args[0]),
                    std::move(args[1])
                );
            } else if (methodName == "random" && args.size() == 2) {
                return std::make_unique<matrix::TensorRandomExprASTImpl>(
                    std::move(args[0]),
                    std::move(args[1])
                );
            } else if (methodName == "create" && args.size() == 3) {
                // 前两个参数是维度，第三个参数是初始化列表
                return std::make_unique<matrix::TensorCreateExprASTImpl>(
                    std::move(args[0]),
                    std::move(args[1]),
                    std::move(args[2])
                );
            }
        }
    } else if (funcName == "matmul") {
        if (args.size() == 2) {
            return std::make_unique<matrix::MatmulExprASTImpl>(
                std::move(args[0]),
                std::move(args[1])
            );
        }
    } else if (funcName == "print") {
        if (!args.empty()) {
            // 如果参数已经是一个 PrintExprASTImpl，直接返回它
            if (auto* printExpr = dynamic_cast<matrix::PrintExprASTImpl*>(args[0].get())) {
                return std::move(args[0]);
            }
            return std::make_unique<matrix::PrintExprASTImpl>(std::move(args[0]));
        }
        return std::make_unique<matrix::PrintExprASTImpl>(
            std::make_unique<matrix::StringExprASTImpl>("")
        );
    }
    
    // 普通函数调用
    return std::make_unique<matrix::CallExprASTImpl>(
        funcName,
        std::move(args)
    );
}

std::unique_ptr<matrix::ExprAST> BoasASTConverter::visitModule(const ModuleNode* node) {
    std::vector<std::unique_ptr<matrix::ExprAST>> body;
    
    // 处理所有语句
    for (const auto& stmt : node->getBody()) {
        if (auto expr = stmt->accept(this)) {
            body.push_back(std::move(expr));
        }
    }
    
    return std::make_unique<matrix::ModuleASTImpl>(std::move(body));
}

std::unique_ptr<matrix::ExprAST> BoasASTConverter::visitFunctionDef(const FunctionDefNode* node) {
    std::vector<std::unique_ptr<matrix::ExprAST>> body;
    
    // 处理函数体中的所有语句
    for (const auto& stmt : node->getBody()) {
        if (auto expr = stmt->accept(this)) {
            body.push_back(std::move(expr));
        }
    }
    
    return std::make_unique<matrix::FunctionASTImpl>(
        node->getName(),
        node->getArgs(),
        std::move(body)
    );
}

std::unique_ptr<matrix::ExprAST> BoasASTConverter::visitFor(const ForNode* node) {
    auto target = std::make_unique<matrix::VariableExprASTImpl>(node->getTarget());
    auto iter = node->getIter()->accept(this);
    std::vector<std::unique_ptr<matrix::ExprAST>> body;
    
    // 处理循环体中的所有语句
    for (const auto& stmt : node->getBody()) {
        if (auto expr = stmt->accept(this)) {
            body.push_back(std::move(expr));
        }
    }
    
    return std::make_unique<matrix::ForExprASTImpl>(
        std::move(target),
        std::move(iter),
        std::move(body)
    );
}

std::unique_ptr<matrix::ExprAST> BoasASTConverter::visitReturn(const ReturnNode* node) {
    auto value = node->getValue()->accept(this);
    return std::make_unique<matrix::ReturnASTImpl>(std::move(value));
}

std::unique_ptr<matrix::ExprAST> BoasASTConverter::visitString(const StringNode* node) {
    return std::make_unique<matrix::StringExprASTImpl>(node->getValue());
}

} // namespace python
} // namespace boas 