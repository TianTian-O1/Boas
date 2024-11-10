#ifndef BOAS_NUMBER_AST_H
#define BOAS_NUMBER_AST_H

#include "boas/ast/ast.h"
#include <cstdint>

namespace boas {

class NumberAST : public AST {
public:
    static bool classof(const AST* ast) {
        return ast->getKind() == ASTKind::Number;
    }
    
    ASTKind getKind() const override {
        return ASTKind::Number;
    }
    
    explicit NumberAST(int64_t value);
    
    std::unique_ptr<AST> clone() const override;
    bool operator==(const AST& other) const override;
    
    int64_t getValue() const;
    
private:
    int64_t value_;
};

} // namespace boas

#endif // BOAS_NUMBER_AST_H