#ifndef BOAS_PRINT_AST_H
#define BOAS_PRINT_AST_H

#include "boas/ast/ast.h"
#include "llvm/Support/Casting.h"
#include <memory>

namespace boas {

class PrintAST : public AST {
public:
    static bool classof(const AST* ast) {
        return ast->getKind() == ASTKind::Print;
    }
    
    ASTKind getKind() const override {
        return ASTKind::Print;
    }
    
    explicit PrintAST(std::unique_ptr<AST> value) 
        : value_(std::move(value)) {}
    
    std::unique_ptr<AST> clone() const override {
        return std::make_unique<PrintAST>(value_->clone());
    }
    
    bool operator==(const AST& other) const override {
        if (auto* printAST = llvm::dyn_cast<PrintAST>(&other)) {
            return *value_ == *printAST->value_;
        }
        return false;
    }
    
    const AST* getValue() const { return value_.get(); }
    
private:
    std::unique_ptr<AST> value_;
};

} // namespace boas

#endif // BOAS_PRINT_AST_H
