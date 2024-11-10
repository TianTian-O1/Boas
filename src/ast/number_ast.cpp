#include "boas/ast/number_ast.h"

namespace boas {

NumberAST::NumberAST(int64_t value) : value_(value) {}

std::unique_ptr<AST> NumberAST::clone() const {
    return std::make_unique<NumberAST>(value_);
}

bool NumberAST::operator==(const AST& other) const {
    if (const auto* numberAst = dynamic_cast<const NumberAST*>(&other)) {
        return value_ == numberAst->value_;
    }
    return false;
}

int64_t NumberAST::getValue() const {
    return value_;
}

} // namespace boas