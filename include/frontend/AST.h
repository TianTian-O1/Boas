#ifndef MATRIX_AST_H
#define MATRIX_AST_H

#include <string>
#include <vector>
#include <memory>
#include <iostream>

namespace matrix {

// 基类
class ExprAST {
protected:
    ExprAST* parent = nullptr;
    
public:
    virtual ~ExprAST() = default;
    
    void setParent(ExprAST* p) { parent = p; }
    ExprAST* getParent() const { return parent; }
    
    enum Kind {
        Number,
        Variable,
        Binary,
        Call,
        Function,
        Import,
        Tensor,
        Array,
        Member,
        Print,
        Assignment,
        TensorCreate,
        Matmul,
        TensorRandom,  // Add this new kind
        TimeCall,
        List,
        ListIndex
    };
    
    virtual Kind getKind() const = 0;
    virtual void dump(int indent = 0) const = 0;
    
    // Add static classof method for LLVM RTTI
    static bool classof(const ExprAST* expr) {
        return true;  // Base class accepts all
    }
    
protected:
    void printIndent(int level) const {
        for (int i = 0; i < level; ++i) std::cout << "  ";
    }
};


class TimeCallExprAST : public ExprAST {
    std::string func_name;  // 函数名（如 "now"）
public:
    TimeCallExprAST(const std::string& name) : func_name(name) {}
    
    const std::string& getFuncName() const { return func_name; }
    
    void dump(int indent = 0) const override {
        printIndent(indent);
        std::cout << "time." << func_name << "()";
    }
    
    Kind getKind() const override { return Kind::TimeCall; }
    
    static bool classof(const ExprAST* expr) {
        return expr->getKind() == Kind::TimeCall;
    }
};


// 数组表达式
class ArrayExprAST : public ExprAST {
    std::vector<std::unique_ptr<ExprAST>> elements;
public:
    ArrayExprAST(std::vector<std::unique_ptr<ExprAST>> elements)
        : elements(std::move(elements)) {}
    
    void dump(int indent = 0) const override {
        printIndent(indent);
        std::cout << "[";
        for (size_t i = 0; i < elements.size(); ++i) {
            if (i > 0) std::cout << ", ";
            elements[i]->dump(0);
        }
        std::cout << "]";
    }

    const std::vector<std::unique_ptr<ExprAST>>& getElements() const { return elements; }

    std::vector<std::unique_ptr<ExprAST>> takeElements() {
        return std::move(elements);
    }

    Kind getKind() const override { return Kind::Array; }
};

// 张量表达式
class TensorExprAST : public ExprAST {
    std::vector<std::unique_ptr<ExprAST>> elements;
public:
    TensorExprAST(std::vector<std::unique_ptr<ExprAST>> elements)
        : elements(std::move(elements)) {}
    
    TensorExprAST(ArrayExprAST& array)
        : elements(array.takeElements()) {}
    
    void dump(int indent = 0) const override {
        printIndent(indent);
        std::cout << "tensor(";
        for (size_t i = 0; i < elements.size(); ++i) {
            if (i > 0) std::cout << ", ";
            elements[i]->dump(0);
        }
        std::cout << ")";
    }

    const std::vector<std::unique_ptr<ExprAST>>& getElements() const { return elements; }

    Kind getKind() const override { return Kind::Tensor; }
};

// 矩阵乘法表达式
class MatmulExprAST : public ExprAST {
    std::unique_ptr<ExprAST> lhs, rhs;
public:
    MatmulExprAST(std::unique_ptr<ExprAST> lhs,
                  std::unique_ptr<ExprAST> rhs)
        : lhs(std::move(lhs)), rhs(std::move(rhs)) {}
    
    // Const getters
    const ExprAST* getLHS() const { return lhs.get(); }
    const ExprAST* getRHS() const { return rhs.get(); }
    
    // Non-const getters
    ExprAST* getLHS() { return lhs.get(); }
    ExprAST* getRHS() { return rhs.get(); }

    void dump(int indent = 0) const override {
        printIndent(indent);
        std::cout << "matmul(";
        lhs->dump(0);
        std::cout << ", ";
        rhs->dump(0);
        std::cout << ")";
    }

    Kind getKind() const override { return Kind::Matmul; }

    static bool classof(const ExprAST* expr) {
        return expr->getKind() == Kind::Matmul;
    }
};

// 变量表达式
class VariableExprAST : public ExprAST {
    std::string name;
public:
    VariableExprAST(const std::string &name) : name(name) {}
    const std::string &getName() const { return name; }
    
    void dump(int indent = 0) const override {
        printIndent(indent);
        std::cout << name;
    }

    Kind getKind() const override { return Kind::Variable; }

    static bool classof(const ExprAST* expr) {
        return expr->getKind() == Kind::Variable;
    }
};

// 数字表达式
class NumberExprAST : public ExprAST {
    double Val;
public:
    NumberExprAST(double Val) : Val(Val) {}
    double getValue() const { return Val; }
    
    void dump(int indent = 0) const override {
        printIndent(indent);
        std::cout << Val;
    }
    
    Kind getKind() const override { return Kind::Number; }

    static bool classof(const ExprAST* expr) {
        return expr->getKind() == Kind::Number;
    }
};

// 赋值表达式
class AssignmentExprAST : public ExprAST {
    std::string Name;
    std::unique_ptr<ExprAST> Value;
public:
    AssignmentExprAST(const std::string &Name, std::unique_ptr<ExprAST> Value)
        : Name(Name), Value(std::move(Value)) {}
    
    const std::string &getName() const { return Name; }
    const ExprAST* getValue() const { return Value.get(); }
    
    void dump(int indent = 0) const override {
        printIndent(indent);
        std::cout << "Assignment " << Name << " = ";
        Value->dump(0);
    }
    
    Kind getKind() const override { return Kind::Assignment; }

    static bool classof(const ExprAST* expr) {
        return expr->getKind() == Kind::Assignment;
    }
};

// 导入语句
class ImportAST : public ExprAST {
    std::string module_name;
    std::string func_name;  // 可选，用于 from-import
public:
    ImportAST(std::string module) : module_name(std::move(module)) {}
    ImportAST(std::string module, std::string func) 
        : module_name(std::move(module)), func_name(std::move(func)) {}
    
    void dump(int indent = 0) const override {
        printIndent(indent);
        if (func_name.empty()) {
            std::cout << "import " << module_name;
        } else {
            std::cout << "from " << module_name << " import " << func_name;
        }
    }

    const std::string& getModuleName() const { return module_name; }

    Kind getKind() const override { return Kind::Import; }
};

// 函数调用
class CallExprAST : public ExprAST {
    std::string callee_;
    std::vector<std::unique_ptr<ExprAST>> args_;

public:
    CallExprAST(const std::string &callee,
                std::vector<std::unique_ptr<ExprAST>> args)
        : callee_(callee), args_(std::move(args)) {}

    const std::string& getCallee() const { return callee_; }
    const std::vector<std::unique_ptr<ExprAST>>& getArgs() const { return args_; }

    Kind getKind() const override { return Kind::Call; }

    static bool classof(const ExprAST* expr) {
        return expr->getKind() == Kind::Call;
    }

    void dump(int indent = 0) const override {
        printIndent(indent);
        std::cout << "Call " << callee_ << "(";
        for (const auto& arg : args_) {
            arg->dump(0);
            std::cout << " ";
        }
        std::cout << ")";
    }
};

// 函数定义
class FunctionAST : public ExprAST {
    std::string name;
    std::vector<std::string> args;
    std::vector<std::unique_ptr<ExprAST>> body;
public:
    FunctionAST(std::string name, std::vector<std::string> args,
                std::vector<std::unique_ptr<ExprAST>> body)
        : name(std::move(name)), args(std::move(args)), body(std::move(body)) {}
    
    void dump(int indent = 0) const override {
        printIndent(indent);
        std::cout << "def " << name << "(";
        for (size_t i = 0; i < args.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << args[i];
        }
        std::cout << "):\n";
        for (const auto& stmt : body) {
            stmt->dump(indent + 1);
            std::cout << "\n";
        }
    }

    const std::string& getName() const { return name; }
    const std::vector<std::string>& getArgs() const { return args; }
    const std::vector<std::unique_ptr<ExprAST>>& getBody() const { return body; }

    Kind getKind() const override { return Kind::Function; }
};

// Member access expression (e.g., tensor.matmul)
class MemberExprAST : public ExprAST {
    std::unique_ptr<ExprAST> object;
    std::string member;
public:
    MemberExprAST(std::unique_ptr<ExprAST> object, std::string member)
        : object(std::move(object)), member(member) {}
    
    void dump(int indent = 0) const override {
        printIndent(indent);
        std::cout << "(member ";
        object->dump(0);
        std::cout << " " << member << ")";
    }

    Kind getKind() const override { return Kind::Member; }
};

class PrintExprAST : public ExprAST {
    std::unique_ptr<ExprAST> Value;
public:
    PrintExprAST(std::unique_ptr<ExprAST> Value)
        : Value(std::move(Value)) {}
    
    const ExprAST* getValue() const { return Value.get(); }
    
    void dump(int indent = 0) const override {
        printIndent(indent);
        std::cout << "print ";
        Value->dump(0);
    }
    
    Kind getKind() const override { return Kind::Print; }
};

class TensorCreateExprAST : public ExprAST {
    std::unique_ptr<ExprAST> rows_;
    std::unique_ptr<ExprAST> cols_;
    std::vector<std::unique_ptr<ExprAST>> values_;
public:
    TensorCreateExprAST(std::unique_ptr<ExprAST> rows,
                        std::unique_ptr<ExprAST> cols,
                        std::vector<std::unique_ptr<ExprAST>> values)
        : rows_(std::move(rows))
        , cols_(std::move(cols))
        , values_(std::move(values)) {}
    
    void dump(int indent = 0) const override {
        printIndent(indent);
        std::cout << "tensor.create(";
        rows_->dump(0);
        std::cout << ", ";
        cols_->dump(0);
        std::cout << "){";
        for (size_t i = 0; i < values_.size(); ++i) {
            if (i > 0) std::cout << ", ";
            values_[i]->dump(0);
        }
        std::cout << "}";
    }
    
    const ExprAST* getRows() const { return rows_.get(); }
    const ExprAST* getCols() const { return cols_.get(); }
    const std::vector<std::unique_ptr<ExprAST>>& getValues() const { return values_; }
    
    Kind getKind() const override { return Kind::TensorCreate; }
};


class TensorRandomExprAST : public ExprAST {
    std::unique_ptr<ExprAST> rows_;
    std::unique_ptr<ExprAST> cols_;
public:
    TensorRandomExprAST(std::unique_ptr<ExprAST> rows,
                        std::unique_ptr<ExprAST> cols)
        : rows_(std::move(rows))
        , cols_(std::move(cols)) {}
    
    void dump(int indent = 0) const override {
        printIndent(indent);
        std::cout << "tensor.random(";
        rows_->dump(0);
        std::cout << ", ";
        cols_->dump(0);
        std::cout << ")";
    }
    
    const ExprAST* getRows() const { return rows_.get(); }
    const ExprAST* getCols() const { return cols_.get(); }
    
    Kind getKind() const override { return Kind::TensorRandom; }
};

class BinaryExprAST : public ExprAST {
    std::unique_ptr<ExprAST> lhs_;
    std::unique_ptr<ExprAST> rhs_;
    std::string op_;
public:
    BinaryExprAST(std::unique_ptr<ExprAST> lhs,
                  std::unique_ptr<ExprAST> rhs,
                  const std::string& op)
        : lhs_(std::move(lhs))
        , rhs_(std::move(rhs))
        , op_(op) {}
    
    const ExprAST* getLHS() const { return lhs_.get(); }
    const ExprAST* getRHS() const { return rhs_.get(); }
    const std::string& getOp() const { return op_; }
    
    void dump(int indent = 0) const override {
        printIndent(indent);
        std::cout << "(" << op_ << " ";
        lhs_->dump(0);
        std::cout << " ";
        rhs_->dump(0);
        std::cout << ")";
    }
    
    Kind getKind() const override { return Kind::Binary; }
};

// 列表表达式
class ListExprAST : public ExprAST {
    std::vector<std::unique_ptr<ExprAST>> elements;
public:
    ListExprAST(std::vector<std::unique_ptr<ExprAST>> elements)
        : elements(std::move(elements)) {}
    
    void dump(int indent = 0) const override {
        printIndent(indent);
        std::cout << "[";
        for (size_t i = 0; i < elements.size(); ++i) {
            if (i > 0) std::cout << ", ";
            elements[i]->dump(0);
        }
        std::cout << "]";
    }

    const std::vector<std::unique_ptr<ExprAST>>& getElements() const { 
        return elements; 
    }

    Kind getKind() const override { return Kind::List; }
    
    static bool classof(const ExprAST* expr) {
        return expr->getKind() == Kind::List;
    }
};

// 列表索引访问表达式
class ListIndexExprAST : public ExprAST {
    std::unique_ptr<ExprAST> list;
    std::unique_ptr<ExprAST> index;
public:
    ListIndexExprAST(std::unique_ptr<ExprAST> list, 
                     std::unique_ptr<ExprAST> index)
        : list(std::move(list)), index(std::move(index)) {}
    
    const ExprAST* getList() const { return list.get(); }
    const ExprAST* getIndex() const { return index.get(); }
    
    void dump(int indent = 0) const override {
        printIndent(indent);
        list->dump(0);
        std::cout << "[";
        index->dump(0);
        std::cout << "]";
    }
    
    Kind getKind() const override { return Kind::ListIndex; }
    
    static bool classof(const ExprAST* expr) {
        return expr->getKind() == Kind::ListIndex;
    }
};

} // namespace matrix

#endif // MATRIX_AST_H
