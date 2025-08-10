#ifndef MATRIX_AST_IMPL_H
#define MATRIX_AST_IMPL_H

#include "frontend/AST.h"
#include "frontend/ModuleAST.h"
#include "frontend/FunctionAST.h"
#include "frontend/ControlFlowAST.h"
#include "frontend/ComprehensionAST.h"
#include "frontend/BasicAST.h"

namespace matrix {

class VariableExprASTImpl : public VariableExprAST {
    std::string name_;
public:
    VariableExprASTImpl(const std::string &name) : name_(name) {}
    const std::string &getName() const override { return name_; }
    ExprAST* clone() const override {
        return new VariableExprASTImpl(name_);
    }
};

class MatmulExprASTImpl : public MatmulExprAST {
    std::unique_ptr<ExprAST> lhs_, rhs_;
public:
    MatmulExprASTImpl(std::unique_ptr<ExprAST> lhs, std::unique_ptr<ExprAST> rhs)
        : lhs_(std::move(lhs)), rhs_(std::move(rhs)) {}
    ExprAST &getLHS() const override { return *lhs_; }
    ExprAST &getRHS() const override { return *rhs_; }
    ExprAST* clone() const override {
        return new MatmulExprASTImpl(
            std::unique_ptr<ExprAST>(lhs_->clone()),
            std::unique_ptr<ExprAST>(rhs_->clone())
        );
    }
};

class NumberExprASTImpl : public NumberExprAST {
    double val_;
public:
    NumberExprASTImpl(double val) : val_(val) {}
    double getValue() const override { return val_; }
    ExprAST* clone() const override {
        return new NumberExprASTImpl(val_);
    }
};

class BinaryExprASTImpl : public BinaryExprAST {
    std::string op_;
    std::unique_ptr<ExprAST> lhs_, rhs_;
public:
    BinaryExprASTImpl(const std::string &op,
                      std::unique_ptr<ExprAST> lhs,
                      std::unique_ptr<ExprAST> rhs)
        : op_(op), lhs_(std::move(lhs)), rhs_(std::move(rhs)) {}
    const std::string &getOp() const override { return op_; }
    ExprAST &getLHS() const override { return *lhs_; }
    ExprAST &getRHS() const override { return *rhs_; }
    ExprAST* clone() const override {
        return new BinaryExprASTImpl(
            op_,
            std::unique_ptr<ExprAST>(lhs_->clone()),
            std::unique_ptr<ExprAST>(rhs_->clone())
        );
    }
};

class CallExprASTImpl : public CallExprAST {
    std::string callee_;
    std::vector<std::unique_ptr<ExprAST>> args_;
public:
    CallExprASTImpl(const std::string &callee,
                    std::vector<std::unique_ptr<ExprAST>> args)
        : callee_(callee), args_(std::move(args)) {}
    const std::string &getCallee() const override { return callee_; }
    const std::vector<std::unique_ptr<ExprAST>> &getArgs() const override { return args_; }
    ExprAST* clone() const override {
        std::vector<std::unique_ptr<ExprAST>> clonedArgs;
        for (const auto& arg : args_) {
            clonedArgs.push_back(std::unique_ptr<ExprAST>(arg->clone()));
        }
        return new CallExprASTImpl(callee_, std::move(clonedArgs));
    }
};

class AssignmentExprASTImpl : public AssignmentExprAST {
    std::string name_;
    std::unique_ptr<ExprAST> value_;
public:
    AssignmentExprASTImpl(const std::string &name, std::unique_ptr<ExprAST> value)
        : name_(name), value_(std::move(value)) {}
    const std::string &getName() const override { return name_; }
    ExprAST &getValue() const override { return *value_; }
    ExprAST* clone() const override {
        return new AssignmentExprASTImpl(
            name_,
            std::unique_ptr<ExprAST>(value_->clone())
        );
    }
};

class ModuleASTImpl : public ModuleAST {
    std::vector<std::unique_ptr<ExprAST>> body_;
public:
    ModuleASTImpl(std::vector<std::unique_ptr<ExprAST>> body)
        : body_(std::move(body)) {}
    const std::vector<std::unique_ptr<ExprAST>>& getBody() const override { return body_; }
    ExprAST* clone() const override {
        std::vector<std::unique_ptr<ExprAST>> clonedBody;
        for (const auto& stmt : body_) {
            clonedBody.push_back(std::unique_ptr<ExprAST>(stmt->clone()));
        }
        return new ModuleASTImpl(std::move(clonedBody));
    }
};

class FunctionASTImpl : public FunctionAST {
    std::string name_;
    std::vector<std::unique_ptr<ArgumentAST>> args_;
    std::vector<std::unique_ptr<ExprAST>> body_;
    std::string returnType_;
public:
    FunctionASTImpl(const std::string& name,
                   std::vector<std::unique_ptr<ArgumentAST>> args,
                   std::vector<std::unique_ptr<ExprAST>> body,
                   const std::string& returnType = "")
        : name_(name), args_(std::move(args)), body_(std::move(body)), returnType_(returnType) {}
    const std::string& getName() const override { return name_; }
    const std::vector<std::unique_ptr<ArgumentAST>>& getArgs() const override { return args_; }
    const std::vector<std::unique_ptr<ExprAST>>& getBody() const override { return body_; }
    const std::string& getReturnType() const override { return returnType_; }
    ExprAST* clone() const override {
        std::vector<std::unique_ptr<ArgumentAST>> clonedArgs;
        for (const auto& arg : args_) {
            clonedArgs.push_back(std::unique_ptr<ArgumentAST>(static_cast<ArgumentAST*>(arg->clone())));
        }
        std::vector<std::unique_ptr<ExprAST>> clonedBody;
        for (const auto& stmt : body_) {
            clonedBody.push_back(std::unique_ptr<ExprAST>(stmt->clone()));
        }
        return new FunctionASTImpl(name_, std::move(clonedArgs), std::move(clonedBody), returnType_);
    }
};

class ReturnASTImpl : public ReturnAST {
    std::unique_ptr<ExprAST> value_;
public:
    ReturnASTImpl(std::unique_ptr<ExprAST> value)
        : value_(std::move(value)) {}
    const ExprAST* getValue() const override { return value_.get(); }
    ExprAST* clone() const override {
        return new ReturnASTImpl(
            std::unique_ptr<ExprAST>(value_ ? value_->clone() : nullptr)
        );
    }
};

class ForExprASTImpl : public ForExprAST {
    std::unique_ptr<ExprAST> target_, iter_;
    std::vector<std::unique_ptr<ExprAST>> body_;
public:
    ForExprASTImpl(std::unique_ptr<ExprAST> target,
                   std::unique_ptr<ExprAST> iter,
                   std::vector<std::unique_ptr<ExprAST>> body)
        : target_(std::move(target)), iter_(std::move(iter)), body_(std::move(body)) {}
    const ExprAST* getTarget() const override { return target_.get(); }
    const ExprAST* getIter() const override { return iter_.get(); }
    const std::vector<std::unique_ptr<ExprAST>>& getBody() const override { return body_; }
    ExprAST* clone() const override {
        std::vector<std::unique_ptr<ExprAST>> clonedBody;
        for (const auto& stmt : body_) {
            clonedBody.push_back(std::unique_ptr<ExprAST>(stmt->clone()));
        }
        return new ForExprASTImpl(
            std::unique_ptr<ExprAST>(target_->clone()),
            std::unique_ptr<ExprAST>(iter_->clone()),
            std::move(clonedBody)
        );
    }
};

class WhileExprASTImpl : public WhileExprAST {
    std::unique_ptr<ExprAST> test_;
    std::vector<std::unique_ptr<ExprAST>> body_;
public:
    WhileExprASTImpl(std::unique_ptr<ExprAST> test,
                     std::vector<std::unique_ptr<ExprAST>> body)
        : test_(std::move(test)), body_(std::move(body)) {}
    const ExprAST* getTest() const override { return test_.get(); }
    const std::vector<std::unique_ptr<ExprAST>>& getBody() const override { return body_; }
    ExprAST* clone() const override {
        std::vector<std::unique_ptr<ExprAST>> clonedBody;
        for (const auto& stmt : body_) {
            clonedBody.push_back(std::unique_ptr<ExprAST>(stmt->clone()));
        }
        return new WhileExprASTImpl(
            std::unique_ptr<ExprAST>(test_->clone()),
            std::move(clonedBody)
        );
    }
};

class ListCompASTImpl : public ListCompAST {
    std::unique_ptr<ExprAST> elt_;
    std::vector<std::unique_ptr<ExprAST>> generators_;
public:
    ListCompASTImpl(std::unique_ptr<ExprAST> elt,
                    std::vector<std::unique_ptr<ExprAST>> generators)
        : elt_(std::move(elt)), generators_(std::move(generators)) {}
    const ExprAST* getElt() const override { return elt_.get(); }
    const std::vector<std::unique_ptr<ExprAST>>& getGenerators() const override { return generators_; }
    ExprAST* clone() const override {
        std::vector<std::unique_ptr<ExprAST>> clonedGenerators;
        for (const auto& gen : generators_) {
            clonedGenerators.push_back(std::unique_ptr<ExprAST>(gen->clone()));
        }
        return new ListCompASTImpl(
            std::unique_ptr<ExprAST>(elt_->clone()),
            std::move(clonedGenerators)
        );
    }
};

class CompForASTImpl : public CompForAST {
    std::unique_ptr<ExprAST> target_, iter_;
    std::vector<std::unique_ptr<ExprAST>> ifs_;
public:
    CompForASTImpl(std::unique_ptr<ExprAST> target,
                   std::unique_ptr<ExprAST> iter,
                   std::vector<std::unique_ptr<ExprAST>> ifs)
        : target_(std::move(target)), iter_(std::move(iter)), ifs_(std::move(ifs)) {}
    const ExprAST* getTarget() const override { return target_.get(); }
    const ExprAST* getIter() const override { return iter_.get(); }
    const std::vector<std::unique_ptr<ExprAST>>& getIfs() const override { return ifs_; }
    ExprAST* clone() const override {
        std::vector<std::unique_ptr<ExprAST>> clonedIfs;
        for (const auto& if_expr : ifs_) {
            clonedIfs.push_back(std::unique_ptr<ExprAST>(if_expr->clone()));
        }
        return new CompForASTImpl(
            std::unique_ptr<ExprAST>(target_->clone()),
            std::unique_ptr<ExprAST>(iter_->clone()),
            std::move(clonedIfs)
        );
    }
};

class ListExprASTImpl : public ListExprAST {
    std::vector<std::unique_ptr<ExprAST>> elements_;
public:
    ListExprASTImpl(std::vector<std::unique_ptr<ExprAST>> elements)
        : elements_(std::move(elements)) {}
    const std::vector<std::unique_ptr<ExprAST>>& getElements() const override { return elements_; }
    ExprAST* clone() const override {
        std::vector<std::unique_ptr<ExprAST>> clonedElements;
        for (const auto& element : elements_) {
            clonedElements.push_back(std::unique_ptr<ExprAST>(element->clone()));
        }
        return new ListExprASTImpl(std::move(clonedElements));
    }
};

class SubscriptExprASTImpl : public SubscriptExprAST {
    std::unique_ptr<ExprAST> value_, slice_;
public:
    SubscriptExprASTImpl(std::unique_ptr<ExprAST> value,
                         std::unique_ptr<ExprAST> slice)
        : value_(std::move(value)), slice_(std::move(slice)) {}
    const ExprAST* getValue() const override { return value_.get(); }
    const ExprAST* getSlice() const override { return slice_.get(); }
    ExprAST* clone() const override {
        return new SubscriptExprASTImpl(
            std::unique_ptr<ExprAST>(value_->clone()),
            std::unique_ptr<ExprAST>(slice_->clone())
        );
    }
};

class AttributeExprASTImpl : public AttributeExprAST {
    std::unique_ptr<ExprAST> value_;
    std::string attr_;
public:
    AttributeExprASTImpl(std::unique_ptr<ExprAST> value,
                        const std::string& attr)
        : value_(std::move(value)), attr_(attr) {}
    const ExprAST* getValue() const override { return value_.get(); }
    const std::string& getAttr() const override { return attr_; }
    ExprAST* clone() const override {
        return new AttributeExprASTImpl(
            std::unique_ptr<ExprAST>(value_->clone()),
            attr_
        );
    }
};

class StringExprASTImpl : public StringExprAST {
    std::string value_;
public:
    StringExprASTImpl(const std::string& value) : value_(value) {}
    const std::string& getValue() const override { return value_; }
    ExprAST* clone() const override {
        return new StringExprASTImpl(value_);
    }
};

} // namespace matrix

#endif // MATRIX_AST_IMPL_H 