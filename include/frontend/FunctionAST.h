#ifndef BOAS_FUNCTION_AST_H
#define BOAS_FUNCTION_AST_H

#include "AST.h"
#include <vector>
#include <memory>

namespace boas {

class ArgumentAST : public matrix::ExprAST {
    std::string name;
public:
    ArgumentAST(const std::string& name) : name(name) {}
    
    const std::string& getName() const { return name; }
    
    void dump(int indent = 0) const override {
        printIndent(indent);
        std::cout << "Argument: " << name;
    }
    
    Kind getKind() const override { return Kind::Argument; }
    
    matrix::ExprAST* clone() const override {
        return new ArgumentAST(name);
    }
};

class FunctionAST : public matrix::ExprAST {
    std::string name;
    std::vector<std::unique_ptr<ArgumentAST>> args;
    std::vector<std::unique_ptr<matrix::ExprAST>> body;
    std::string returnType;
public:
    FunctionAST(const std::string& name,
                std::vector<std::unique_ptr<ArgumentAST>> args,
                std::vector<std::unique_ptr<matrix::ExprAST>> body,
                const std::string& returnType = "")
        : name(name)
        , args(std::move(args))
        , body(std::move(body))
        , returnType(returnType) {}
    
    const std::string& getName() const { return name; }
    const std::vector<std::unique_ptr<ArgumentAST>>& getArgs() const { return args; }
    const std::vector<std::unique_ptr<matrix::ExprAST>>& getBody() const { return body; }
    const std::string& getReturnType() const { return returnType; }
    
    void dump(int indent = 0) const override {
        printIndent(indent);
        std::cout << "Function " << name << "(";
        for (size_t i = 0; i < args.size(); ++i) {
            if (i > 0) std::cout << ", ";
            args[i]->dump(0);
        }
        std::cout << ")";
        if (!returnType.empty()) {
            std::cout << " -> " << returnType;
        }
        std::cout << ":\n";
        for (const auto& stmt : body) {
            stmt->dump(indent + 1);
            std::cout << "\n";
        }
    }
    
    Kind getKind() const override { return Kind::Function; }
    
    matrix::ExprAST* clone() const override {
        std::vector<std::unique_ptr<ArgumentAST>> clonedArgs;
        for (const auto& arg : args) {
            clonedArgs.push_back(std::unique_ptr<ArgumentAST>(
                static_cast<ArgumentAST*>(arg->clone())));
        }
        
        std::vector<std::unique_ptr<matrix::ExprAST>> clonedBody;
        for (const auto& stmt : body) {
            clonedBody.push_back(std::unique_ptr<matrix::ExprAST>(stmt->clone()));
        }
        
        return new FunctionAST(name, std::move(clonedArgs), std::move(clonedBody), returnType);
    }
};

class ReturnAST : public matrix::ExprAST {
    std::unique_ptr<matrix::ExprAST> value;
public:
    ReturnAST(std::unique_ptr<matrix::ExprAST> value)
        : value(std::move(value)) {}
    
    const matrix::ExprAST* getValue() const { return value.get(); }
    
    void dump(int indent = 0) const override {
        printIndent(indent);
        std::cout << "Return ";
        if (value) {
            value->dump(0);
        }
    }
    
    Kind getKind() const override { return Kind::Return; }
    
    matrix::ExprAST* clone() const override {
        return new ReturnAST(
            std::unique_ptr<matrix::ExprAST>(value ? value->clone() : nullptr));
    }
};

} // namespace boas

#endif // BOAS_FUNCTION_AST_H 