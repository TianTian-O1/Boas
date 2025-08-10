#include "frontend/SimpleAST.h"
#include <sstream>

namespace matrix {

std::string Literal::toString() const {
    std::stringstream ss;
    std::visit([&ss](const auto& val) { ss << val; }, value_);
    return ss.str();
}

std::string BinaryOp::toString() const {
    static const std::map<OpType, std::string> op_str = {
        {OpType::Add, "+"},
        {OpType::Sub, "-"},
        {OpType::Mul, "*"},
        {OpType::Div, "/"},
        {OpType::Eq, "=="},
        {OpType::Lt, "<"},
        {OpType::Gt, ">"}
    };
    
    std::stringstream ss;
    ss << "(" << left_->toString() << " " 
       << op_str.at(op_) << " " 
       << right_->toString() << ")";
    return ss.str();
}

std::string Call::toString() const {
    std::stringstream ss;
    ss << func_name_ << "(";
    for (size_t i = 0; i < args_.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << args_[i]->toString();
    }
    ss << ")";
    return ss.str();
}

std::string Assignment::toString() const {
    return target_ + " = " + value_->toString();
}

std::string FunctionDef::toString() const {
    std::stringstream ss;
    ss << "def " << name_ << "(";
    for (size_t i = 0; i < params_.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << params_[i];
    }
    ss << "):\n";
    for (const auto& stmt : body_) {
        ss << "    " << stmt->toString() << "\n";
    }
    return ss.str();
}

// ASTBuilder实现
std::unique_ptr<Expression> ASTBuilder::num(int value) {
    return std::make_unique<Literal>(value);
}

std::unique_ptr<Expression> ASTBuilder::num(double value) {
    return std::make_unique<Literal>(value);
}

std::unique_ptr<Expression> ASTBuilder::str(const std::string& value) {
    return std::make_unique<Literal>(value);
}

std::unique_ptr<Expression> ASTBuilder::boolean(bool value) {
    return std::make_unique<Literal>(value);
}

std::unique_ptr<Expression> ASTBuilder::name(const std::string& name) {
    return std::make_unique<Identifier>(name);
}

std::unique_ptr<Expression> ASTBuilder::add(std::unique_ptr<Expression> left,
                                          std::unique_ptr<Expression> right) {
    return std::make_unique<BinaryOp>(std::move(left), 
                                     BinaryOp::OpType::Add, 
                                     std::move(right));
}

std::unique_ptr<Expression> ASTBuilder::sub(std::unique_ptr<Expression> left,
                                          std::unique_ptr<Expression> right) {
    return std::make_unique<BinaryOp>(std::move(left), 
                                     BinaryOp::OpType::Sub, 
                                     std::move(right));
}

std::unique_ptr<Expression> ASTBuilder::mul(std::unique_ptr<Expression> left,
                                          std::unique_ptr<Expression> right) {
    return std::make_unique<BinaryOp>(std::move(left), 
                                     BinaryOp::OpType::Mul, 
                                     std::move(right));
}

std::unique_ptr<Expression> ASTBuilder::div(std::unique_ptr<Expression> left,
                                          std::unique_ptr<Expression> right) {
    return std::make_unique<BinaryOp>(std::move(left), 
                                     BinaryOp::OpType::Div, 
                                     std::move(right));
}

std::unique_ptr<Expression> ASTBuilder::call(const std::string& func_name,
                                           std::vector<std::unique_ptr<Expression>> args) {
    return std::make_unique<Call>(func_name, std::move(args));
}

std::unique_ptr<Statement> ASTBuilder::assign(const std::string& target,
                                            std::unique_ptr<Expression> value) {
    return std::make_unique<Assignment>(target, std::move(value));
}

std::unique_ptr<Statement> ASTBuilder::def(const std::string& name,
                                         const std::vector<std::string>& params,
                                         std::vector<std::unique_ptr<Statement>> body) {
    return std::make_unique<FunctionDef>(name, params, std::move(body));
}

} // namespace matrix 