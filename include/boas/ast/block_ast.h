#ifndef BOAS_BLOCK_AST_H
#define BOAS_BLOCK_AST_H

#include "boas/ast/ast.h"
#include "llvm/Support/Casting.h"
#include <vector>
#include <memory>

namespace boas {

class BlockAST : public AST {
public:
    static bool classof(const AST* ast) {
        return ast->getKind() == ASTKind::Block;
    }
    
    ASTKind getKind() const override {
        return ASTKind::Block;
    }
    
    explicit BlockAST(std::vector<std::unique_ptr<AST>> statements) 
        : statements_(std::move(statements)) {}
    
    std::unique_ptr<AST> clone() const override {
        std::vector<std::unique_ptr<AST>> clonedStatements;
        for (const auto& stmt : statements_) {
            clonedStatements.push_back(stmt->clone());
        }
        return std::make_unique<BlockAST>(std::move(clonedStatements));
    }
    
    bool operator==(const AST& other) const override {
        if (auto* blockAST = llvm::dyn_cast<BlockAST>(&other)) {
            if (statements_.size() != blockAST->statements_.size()) {
                return false;
            }
            for (size_t i = 0; i < statements_.size(); ++i) {
                if (!(*statements_[i] == *blockAST->statements_[i])) {
                    return false;
                }
            }
            return true;
        }
        return false;
    }
    
    const std::vector<std::unique_ptr<AST>>& getStatements() const { 
        return statements_; 
    }
    
private:
    std::vector<std::unique_ptr<AST>> statements_;
};

} // namespace boas

#endif // BOAS_BLOCK_AST_H
