#ifndef BOAS_CONTROL_FLOW_AST_H
#define BOAS_CONTROL_FLOW_AST_H

#include "AST.h"
#include <vector>
#include <memory>

namespace boas {

class ForExprAST : public matrix::ExprAST {
    std::unique_ptr<matrix::ExprAST> target;
    std::unique_ptr<matrix::ExprAST> iter;
    std::vector<std::unique_ptr<matrix::ExprAST>> body;
public:
    ForExprAST(std::unique_ptr<matrix::ExprAST> target,
               std::unique_ptr<matrix::ExprAST> iter,
               std::vector<std::unique_ptr<matrix::ExprAST>> body)
        : target(std::move(target))
        , iter(std::move(iter))
        , body(std::move(body)) {}
    
    const matrix::ExprAST* getTarget() const { return target.get(); }
    const matrix::ExprAST* getIter() const { return iter.get(); }
    const std::vector<std::unique_ptr<matrix::ExprAST>>& getBody() const { return body; }
    
    void dump(int indent = 0) const override {
        printIndent(indent);
        std::cout << "For ";
        target->dump(0);
        std::cout << " in ";
        iter->dump(0);
        std::cout << ":\n";
        for (const auto& stmt : body) {
            stmt->dump(indent + 1);
            std::cout << "\n";
        }
    }
    
    Kind getKind() const override { return Kind::For; }
    
    matrix::ExprAST* clone() const override {
        return new ForExprAST(
            std::unique_ptr<matrix::ExprAST>(target->clone()),
            std::unique_ptr<matrix::ExprAST>(iter->clone()),
            cloneBody());
    }

private:
    std::vector<std::unique_ptr<matrix::ExprAST>> cloneBody() const {
        std::vector<std::unique_ptr<matrix::ExprAST>> clonedBody;
        for (const auto& stmt : body) {
            clonedBody.push_back(std::unique_ptr<matrix::ExprAST>(stmt->clone()));
        }
        return clonedBody;
    }
};

class WhileExprAST : public matrix::ExprAST {
    std::unique_ptr<matrix::ExprAST> test;
    std::vector<std::unique_ptr<matrix::ExprAST>> body;
public:
    WhileExprAST(std::unique_ptr<matrix::ExprAST> test,
                 std::vector<std::unique_ptr<matrix::ExprAST>> body)
        : test(std::move(test))
        , body(std::move(body)) {}
    
    const matrix::ExprAST* getTest() const { return test.get(); }
    const std::vector<std::unique_ptr<matrix::ExprAST>>& getBody() const { return body; }
    
    void dump(int indent = 0) const override {
        printIndent(indent);
        std::cout << "While ";
        test->dump(0);
        std::cout << ":\n";
        for (const auto& stmt : body) {
            stmt->dump(indent + 1);
            std::cout << "\n";
        }
    }
    
    Kind getKind() const override { return Kind::While; }
    
    matrix::ExprAST* clone() const override {
        return new WhileExprAST(
            std::unique_ptr<matrix::ExprAST>(test->clone()),
            cloneBody());
    }

private:
    std::vector<std::unique_ptr<matrix::ExprAST>> cloneBody() const {
        std::vector<std::unique_ptr<matrix::ExprAST>> clonedBody;
        for (const auto& stmt : body) {
            clonedBody.push_back(std::unique_ptr<matrix::ExprAST>(stmt->clone()));
        }
        return clonedBody;
    }
};

} // namespace boas

#endif // BOAS_CONTROL_FLOW_AST_H 