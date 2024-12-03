#pragma once

#include "AST.h"
#include "ModuleAST.h"
#include "FunctionAST.h"
#include "ControlFlowAST.h"
#include "ComprehensionAST.h"
#include "BasicAST.h"
#include <memory>
#include <string>
#include <vector>
#include <sstream>
#include <Python.h>
#include <llvm/Support/raw_ostream.h>

namespace matrix {

using namespace boas;

// 辅助函数：克隆语句列表
inline std::vector<std::unique_ptr<ExprAST>> cloneStatements(const std::vector<std::unique_ptr<ExprAST>>& statements) {
    std::vector<std::unique_ptr<ExprAST>> cloned;
    for (const auto& stmt : statements) {
        cloned.push_back(std::unique_ptr<ExprAST>(stmt->clone()));
    }
    return cloned;
}

class ModuleASTImpl : public matrix::ModuleAST {
    std::vector<std::unique_ptr<ExprAST>> stmts;
public:
    ModuleASTImpl(std::vector<std::unique_ptr<ExprAST>> stmts)
        : ModuleAST(std::move(stmts)) {}
    
    ExprAST* clone() const override {
        std::vector<std::unique_ptr<ExprAST>> newStmts;
        for (const auto& stmt : ModuleAST::getBody()) {
            newStmts.push_back(std::unique_ptr<ExprAST>(stmt->clone()));
        }
        return new ModuleASTImpl(std::move(newStmts));
    }
    
    void dump(int indent = 0) const override {
        ModuleAST::dump(indent);
    }
    
    Kind getKind() const override { return Kind::Module; }
};

class FunctionASTImpl : public matrix::FunctionAST {
public:
    FunctionASTImpl(std::string name, std::vector<std::string> args,
                   std::vector<std::unique_ptr<ExprAST>> body)
        : FunctionAST(std::move(name), std::move(args), std::move(body)) {}
    
    ExprAST* clone() const override {
        std::vector<std::unique_ptr<ExprAST>> newBody;
        for (const auto& stmt : getBody()) {
            newBody.push_back(std::unique_ptr<ExprAST>(stmt->clone()));
        }
        return new FunctionASTImpl(getName(), getArgs(), std::move(newBody));
    }

    void dump(int indent = 0) const override {
        printIndent(indent);
        std::cout << "def " << getName() << "(";
        const auto& args = getArgs();
        for (size_t i = 0; i < args.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << args[i];
        }
        std::cout << "):\n";
        for (const auto& stmt : getBody()) {
            stmt->dump(indent + 8);
            std::cout << "\n";
        }
    }
};

class ReturnASTImpl : public boas::ReturnAST {
public:
    ReturnASTImpl(std::unique_ptr<ExprAST> value)
        : ReturnAST(std::move(value)) {}
    
    ExprAST* clone() const override {
        return new ReturnASTImpl(std::unique_ptr<ExprAST>(getValue()->clone()));
    }

    void dump(int indent = 0) const override {
        printIndent(indent);
        std::cout << "Return ";
        getValue()->dump(0);
    }
};

class NumberExprASTImpl : public matrix::NumberExprAST {
public:
    NumberExprASTImpl(double value) : NumberExprAST(value) {}
    
    ExprAST* clone() const override {
        return new NumberExprASTImpl(getValue());
    }

    void dump(int indent = 0) const override {
        double val = getValue();
        if (val == static_cast<int>(val)) {
            std::cout << static_cast<int>(val);
        } else {
            std::cout << val;
        }
    }
};

class VariableExprASTImpl : public matrix::VariableExprAST {
public:
    VariableExprASTImpl(std::string name)
        : VariableExprAST(name) {}
    
    ExprAST* clone() const override {
        return new VariableExprASTImpl(getName());
    }

    void dump(int indent = 0) const override {
        printIndent(indent);
        std::cout << getName();
    }
};

class BinaryExprASTImpl : public matrix::BinaryExprAST {
public:
    BinaryExprASTImpl(std::unique_ptr<ExprAST> lhs,
                      std::unique_ptr<ExprAST> rhs,
                      std::string op)
        : BinaryExprAST(std::move(lhs), std::move(rhs), op) {}
    
    ExprAST* clone() const override {
        return new BinaryExprASTImpl(
            std::unique_ptr<ExprAST>(getLHS()->clone()),
            std::unique_ptr<ExprAST>(getRHS()->clone()),
            getOp());
    }

    void dump(int indent = 0) const override {
        printIndent(indent);
        getLHS()->dump(0);
        std::cout << " " << getOp() << " ";
        getRHS()->dump(0);
    }
};

class CallExprASTImpl : public matrix::CallExprAST {
public:
    CallExprASTImpl(std::string callee,
                    std::vector<std::unique_ptr<ExprAST>> args)
        : CallExprAST(callee, std::move(args)) {}
    
    ExprAST* clone() const override {
        std::vector<std::unique_ptr<ExprAST>> newArgs;
        for (const auto& arg : getArgs()) {
            newArgs.push_back(std::unique_ptr<ExprAST>(arg->clone()));
        }
        return new CallExprASTImpl(getCallee(), std::move(newArgs));
    }

    void dump(int indent = 0) const override {
        printIndent(indent);
        const auto& args = getArgs();
        
        if (getCallee() == "tensor.create") {
            std::cout << "tensor.create(";
            // 捕获所有参数的输出
            std::vector<std::string> argStrs;
            for (const auto& arg : args) {
                std::stringstream ss;
                std::streambuf* old = std::cout.rdbuf(ss.rdbuf());
                arg->dump(0);
                std::cout.rdbuf(old);
                std::string argStr = ss.str();
                argStr.erase(std::remove(argStr.begin(), argStr.end(), '\n'), argStr.end());
                argStrs.push_back(argStr);
            }
            // 输出所有参数
            for (size_t i = 0; i < argStrs.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << argStrs[i];
            }
            std::cout << ")";
        } else {
            std::cout << getCallee() << "(";
            for (size_t i = 0; i < args.size(); ++i) {
                if (i > 0) std::cout << ", ";
                args[i]->dump(0);
            }
            std::cout << ")";
        }
    }
};

class AssignmentExprASTImpl : public matrix::AssignmentExprAST {
public:
    AssignmentExprASTImpl(std::string name,
                         std::unique_ptr<ExprAST> value)
        : AssignmentExprAST(name, std::move(value)) {}
    
    ExprAST* clone() const override {
        return new AssignmentExprASTImpl(
            getName(),
            std::unique_ptr<ExprAST>(getValue()->clone()));
    }

    void dump(int indent = 0) const override {
        printIndent(indent);
        std::cout << "Assignment " << getName() << " = ";
        
        // 捕获值的输出
        std::stringstream ss;
        std::streambuf* old = std::cout.rdbuf(ss.rdbuf());
        getValue()->dump(0);
        std::cout.rdbuf(old);
        std::string valueStr = ss.str();
        
        // 移除所有换行符
        valueStr.erase(std::remove(valueStr.begin(), valueStr.end(), '\n'), valueStr.end());
        
        // 输出处理后的值
        std::cout << valueStr;
    }
};

class ForExprASTImpl : public boas::ForExprAST {
public:
    ForExprASTImpl(std::unique_ptr<ExprAST> target,
                   std::unique_ptr<ExprAST> iter,
                   std::vector<std::unique_ptr<ExprAST>> body)
        : ForExprAST(std::move(target), std::move(iter), std::move(body)) {}
    
    ExprAST* clone() const override {
        std::vector<std::unique_ptr<ExprAST>> newBody;
        for (const auto& stmt : getBody()) {
            newBody.push_back(std::unique_ptr<ExprAST>(stmt->clone()));
        }
        return new ForExprASTImpl(
            std::unique_ptr<ExprAST>(getTarget()->clone()),
            std::unique_ptr<ExprAST>(getIter()->clone()),
            std::move(newBody));
    }

    void dump(int indent = 0) const override {
        printIndent(indent);
        std::cout << "For ";
        getTarget()->dump(0);
        std::cout << " in ";
        getIter()->dump(0);
        std::cout << ":\n";
        for (const auto& stmt : getBody()) {
            stmt->dump(indent + 4);
            std::cout << "\n";
        }
    }
};

class WhileExprASTImpl : public boas::WhileExprAST {
private:
    std::unique_ptr<matrix::ExprAST> test;
    std::vector<std::unique_ptr<matrix::ExprAST>> body;

public:
    WhileExprASTImpl(std::unique_ptr<matrix::ExprAST> test,
                     std::vector<std::unique_ptr<matrix::ExprAST>> body)
        : WhileExprAST(std::move(test), std::move(body)) {}

    matrix::ExprAST* clone() const override {
        std::vector<std::unique_ptr<matrix::ExprAST>> newBody;
        for (const auto& stmt : getBody()) {
            newBody.push_back(std::unique_ptr<matrix::ExprAST>(stmt->clone()));
        }
        return new WhileExprASTImpl(
            std::unique_ptr<matrix::ExprAST>(getTest()->clone()),
            std::move(newBody)
        );
    }

    Kind getKind() const override { return Kind::While; }
};

class ListCompASTImpl : public boas::ListCompAST {
public:
    ListCompASTImpl(std::unique_ptr<ExprAST> elt,
                    std::vector<std::unique_ptr<ExprAST>> generators)
        : ListCompAST(std::move(elt), std::move(generators)) {}
    
    ExprAST* clone() const override {
        std::vector<std::unique_ptr<ExprAST>> newGens;
        for (const auto& gen : getGenerators()) {
            newGens.push_back(std::unique_ptr<ExprAST>(gen->clone()));
        }
        return new ListCompASTImpl(
            std::unique_ptr<ExprAST>(getElt()->clone()),
            std::move(newGens));
    }
};

class CompForASTImpl : public boas::CompForAST {
public:
    CompForASTImpl(std::unique_ptr<ExprAST> target,
                   std::unique_ptr<ExprAST> iter,
                   std::vector<std::unique_ptr<ExprAST>> ifs)
        : CompForAST(std::move(target), std::move(iter), std::move(ifs)) {}
    
    ExprAST* clone() const override {
        std::vector<std::unique_ptr<ExprAST>> newIfs;
        for (const auto& ifExpr : getIfs()) {
            newIfs.push_back(std::unique_ptr<ExprAST>(ifExpr->clone()));
        }
        return new CompForASTImpl(
            std::unique_ptr<ExprAST>(getTarget()->clone()),
            std::unique_ptr<ExprAST>(getIter()->clone()),
            std::move(newIfs));
    }
};

class SubscriptExprASTImpl : public boas::SubscriptExprAST {
public:
    SubscriptExprASTImpl(std::unique_ptr<ExprAST> value,
                         std::unique_ptr<ExprAST> slice)
        : SubscriptExprAST(std::move(value), std::move(slice)) {}
    
    ExprAST* clone() const override {
        return new SubscriptExprASTImpl(
            std::unique_ptr<ExprAST>(getValue()->clone()),
            std::unique_ptr<ExprAST>(getSlice()->clone()));
    }
};

class AttributeExprASTImpl : public boas::AttributeExprAST {
public:
    AttributeExprASTImpl(std::unique_ptr<ExprAST> value,
                        std::string attr)
        : AttributeExprAST(std::move(value), attr) {}
    
    ExprAST* clone() const override {
        return new AttributeExprASTImpl(
            std::unique_ptr<ExprAST>(getValue()->clone()),
            getAttr());
    }
};

class ImportASTImpl : public matrix::ImportAST {
public:
    ImportASTImpl(std::string moduleName)
        : ImportAST(std::move(moduleName)) {}
    
    ExprAST* clone() const override {
        return new ImportASTImpl(getModuleName());
    }

    void dump(int indent = 0) const override {
        printIndent(indent);
        std::cout << "import " << getModuleName();
    }
};

class StringExprASTImpl : public boas::StringExprAST {
public:
    StringExprASTImpl(std::string value)
        : StringExprAST(value) {}
    
    ExprAST* clone() const override {
        return new StringExprASTImpl(getValue());
    }

    void dump(int indent = 0) const override {
        printIndent(indent);
        std::cout << "\"" << getValue() << "\"";
    }
};

class ConstantExprASTImpl : public boas::ConstantExprAST {
public:
    ConstantExprASTImpl(PyObject* value) : ConstantExprAST(value) {}

    matrix::ExprAST* clone() const override {
        return new ConstantExprASTImpl(getValue());
    }

    Kind getKind() const override { return Kind::Constant; }
};

class MatmulExprASTImpl : public matrix::MatmulExprAST {
public:
    MatmulExprASTImpl(std::unique_ptr<ExprAST> lhs,
                      std::unique_ptr<ExprAST> rhs)
        : MatmulExprAST(std::move(lhs), std::move(rhs)) {}
    
    ExprAST* clone() const override {
        return new MatmulExprASTImpl(
            std::unique_ptr<ExprAST>(getLHS()->clone()),
            std::unique_ptr<ExprAST>(getRHS()->clone()));
    }
};

class TensorCreateExprASTImpl : public matrix::TensorCreateExprAST {
public:
    TensorCreateExprASTImpl(std::unique_ptr<ExprAST> rows,
                           std::unique_ptr<ExprAST> cols,
                           std::unique_ptr<ExprAST> values)
        : TensorCreateExprAST(std::move(rows), std::move(cols), std::move(values)) {}
    
    ExprAST* clone() const override {
        return new TensorCreateExprASTImpl(
            std::unique_ptr<ExprAST>(getRows()->clone()),
            std::unique_ptr<ExprAST>(getCols()->clone()),
            std::unique_ptr<ExprAST>(getValues()->clone())
        );
    }
};

class TensorRandomExprASTImpl : public matrix::TensorRandomExprAST {
public:
    TensorRandomExprASTImpl(std::unique_ptr<matrix::ExprAST> rows,
                           std::unique_ptr<matrix::ExprAST> cols)
        : TensorRandomExprAST(std::move(rows), std::move(cols)) {}

    matrix::ExprAST* clone() const override {
        return new TensorRandomExprASTImpl(
            std::unique_ptr<matrix::ExprAST>(getRows()->clone()),
            std::unique_ptr<matrix::ExprAST>(getCols()->clone())
        );
    }

    Kind getKind() const override { return Kind::TensorRandom; }
};

class PrintExprASTImpl : public matrix::ExprAST {
private:
    std::unique_ptr<ExprAST> value;

public:
    PrintExprASTImpl(std::unique_ptr<ExprAST> val)
        : value(std::move(val)) {}
    
    ExprAST* clone() const override {
        return new PrintExprASTImpl(std::unique_ptr<ExprAST>(value->clone()));
    }

    const ExprAST* getValue() const { return value.get(); }
    ExprAST* getValue() { return value.get(); }
    Kind getKind() const override { return Kind::Print; }

    void dump(int indent = 0) const override {
        printIndent(indent);
        std::cout << "print(";
        value->dump(0);
        std::cout << ")";
    }
};

class ArrayExprASTImpl : public ArrayExprAST {
public:
    ArrayExprASTImpl(std::vector<std::unique_ptr<ExprAST>> elements)
        : ArrayExprAST(std::move(elements)) {}
    
    ExprAST* clone() const override {
        std::vector<std::unique_ptr<ExprAST>> clonedElements;
        for (const auto& element : getElements()) {
            clonedElements.push_back(std::unique_ptr<ExprAST>(element->clone()));
        }
        return new ArrayExprASTImpl(std::move(clonedElements));
    }
};

} // namespace matrix 