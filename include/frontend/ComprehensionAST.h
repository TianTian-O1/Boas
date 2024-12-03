#ifndef BOAS_COMPREHENSION_AST_H
#define BOAS_COMPREHENSION_AST_H

#include "AST.h"
#include <vector>
#include <memory>

namespace boas {

class CompForAST : public matrix::ExprAST {
    std::unique_ptr<matrix::ExprAST> target;
    std::unique_ptr<matrix::ExprAST> iter;
    std::vector<std::unique_ptr<matrix::ExprAST>> ifs;
public:
    CompForAST(std::unique_ptr<matrix::ExprAST> target,
               std::unique_ptr<matrix::ExprAST> iter,
               std::vector<std::unique_ptr<matrix::ExprAST>> ifs)
        : target(std::move(target))
        , iter(std::move(iter))
        , ifs(std::move(ifs)) {}
    
    const matrix::ExprAST* getTarget() const { return target.get(); }
    const matrix::ExprAST* getIter() const { return iter.get(); }
    const std::vector<std::unique_ptr<matrix::ExprAST>>& getIfs() const { return ifs; }
    
    void dump(int indent = 0) const override {
        printIndent(indent);
        std::cout << "CompFor ";
        target->dump(0);
        std::cout << " in ";
        iter->dump(0);
        if (!ifs.empty()) {
            std::cout << " if ";
            for (size_t i = 0; i < ifs.size(); ++i) {
                if (i > 0) std::cout << " and ";
                ifs[i]->dump(0);
            }
        }
    }
    
    Kind getKind() const override { return Kind::CompFor; }
    
    matrix::ExprAST* clone() const override {
        std::vector<std::unique_ptr<matrix::ExprAST>> clonedIfs;
        for (const auto& if_expr : ifs) {
            clonedIfs.push_back(std::unique_ptr<matrix::ExprAST>(if_expr->clone()));
        }
        return new CompForAST(
            std::unique_ptr<matrix::ExprAST>(target->clone()),
            std::unique_ptr<matrix::ExprAST>(iter->clone()),
            std::move(clonedIfs));
    }
};

class ListCompAST : public matrix::ExprAST {
    std::unique_ptr<matrix::ExprAST> elt;
    std::vector<std::unique_ptr<matrix::ExprAST>> generators;
public:
    ListCompAST(std::unique_ptr<matrix::ExprAST> elt,
                std::vector<std::unique_ptr<matrix::ExprAST>> generators)
        : elt(std::move(elt))
        , generators(std::move(generators)) {}
    
    const matrix::ExprAST* getElt() const { return elt.get(); }
    const std::vector<std::unique_ptr<matrix::ExprAST>>& getGenerators() const { return generators; }
    
    void dump(int indent = 0) const override {
        printIndent(indent);
        std::cout << "[";
        elt->dump(0);
        std::cout << " for ";
        for (size_t i = 0; i < generators.size(); ++i) {
            if (i > 0) std::cout << " for ";
            generators[i]->dump(0);
        }
        std::cout << "]";
    }
    
    Kind getKind() const override { return Kind::ListComp; }
    
    matrix::ExprAST* clone() const override {
        std::vector<std::unique_ptr<matrix::ExprAST>> clonedGenerators;
        for (const auto& gen : generators) {
            clonedGenerators.push_back(std::unique_ptr<matrix::ExprAST>(gen->clone()));
        }
        return new ListCompAST(
            std::unique_ptr<matrix::ExprAST>(elt->clone()),
            std::move(clonedGenerators));
    }
};

} // namespace boas

#endif // BOAS_COMPREHENSION_AST_H 