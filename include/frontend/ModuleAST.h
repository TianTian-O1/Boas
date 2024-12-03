#ifndef MATRIX_MODULE_AST_H
#define MATRIX_MODULE_AST_H

#include "frontend/AST.h"
#include <vector>
#include <memory>

namespace matrix {

// 模块表达式，代表一个完整的程序
class ModuleAST : public ExprAST {
    std::vector<std::unique_ptr<ExprAST>> body;
public:
    ModuleAST(std::vector<std::unique_ptr<ExprAST>> body)
        : body(std::move(body)) {}
    
    const std::vector<std::unique_ptr<ExprAST>>& getBody() const { return body; }
    
    void dump(int indent = 0) const override {
        printIndent(indent);
        std::cout << "module:\n";
        for (const auto& stmt : body) {
            stmt->dump(indent + 1);
            std::cout << "\n";
        }
    }
    
    Kind getKind() const override { return Kind::Module; }
    
    ExprAST* clone() const override {
        std::vector<std::unique_ptr<ExprAST>> newBody;
        for (const auto& stmt : body) {
            newBody.push_back(std::unique_ptr<ExprAST>(stmt->clone()));
        }
        return new ModuleAST(std::move(newBody));
    }
};

} // namespace matrix

#endif // MATRIX_MODULE_AST_H 