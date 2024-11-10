#ifndef BOAS_AST_H
#define BOAS_AST_H

#include <memory>

namespace boas {

enum class ASTKind {
    List,
    Number,
    Print,
    Block
};

class AST {
public:
    virtual ~AST() = default;
    
    // Add virtual clone method for polymorphic copying
    virtual std::unique_ptr<AST> clone() const = 0;
    
    // Add virtual equality operator for comparison
    virtual bool operator==(const AST& other) const = 0;
    
    virtual ASTKind getKind() const = 0;
};

} // namespace boas

#endif // BOAS_AST_H
