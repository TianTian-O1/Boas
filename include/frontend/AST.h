#ifndef MATRIX_AST_H
#define MATRIX_AST_H

#include <string>
#include <vector>
#include <memory>
#include <iostream>

namespace matrix {

// 基类
class ExprAST {
public:
    virtual ~ExprAST() = default;
    virtual void dump(int indent = 0) const = 0;
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
        Print
    };
    virtual Kind getKind() const = 0;
protected:
    void printIndent(int level) const {
        for (int i = 0; i < level; ++i) std::cout << "  ";
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
    
    void dump(int indent = 0) const override {
        printIndent(indent);
        std::cout << "matmul(";
        lhs->dump(0);
        std::cout << ", ";
        rhs->dump(0);
        std::cout << ")";
    }

    const ExprAST* getLHS() const { return lhs.get(); }
    const ExprAST* getRHS() const { return rhs.get(); }

    Kind getKind() const override { return Kind::Binary; }
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
};

// 数字表达式
class NumberExprAST : public ExprAST {
    double value;
public:
    NumberExprAST(double value) : value(value) {}
    
    double getValue() const { return value; }
    
    void dump(int indent = 0) const override {
        printIndent(indent);
        std::cout << value;
    }

    Kind getKind() const override { return Kind::Number; }
};

// 赋值表达式
class AssignmentExprAST : public ExprAST {
    std::string name;
    std::unique_ptr<ExprAST> value;
public:
    AssignmentExprAST(std::string name,
                      std::unique_ptr<ExprAST> value)
        : name(std::move(name)), value(std::move(value)) {}
    
    void dump(int indent = 0) const override {
        printIndent(indent);
        std::cout << name << " = ";
        value->dump(0);
    }

    const std::string& getName() const { return name; }
    const ExprAST* getValue() const { return value.get(); }

    Kind getKind() const override { return Kind::Binary; }
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
    std::unique_ptr<ExprAST> object_;  // The object being called on (e.g., tensor in tensor.matmul)
    std::string member_;               // The member function name (e.g., matmul)
    std::vector<std::unique_ptr<ExprAST>> arguments_;  // The function arguments

public:
    CallExprAST(const std::string& member, 
                std::vector<std::unique_ptr<ExprAST>> arguments,
                std::unique_ptr<ExprAST> object)
        : member_(member), 
          arguments_(std::move(arguments)), 
          object_(std::move(object)) {}

    void dump(int indent = 0) const override {
        printIndent(indent);
        std::cout << "Call(";
        if (object_) {
            object_->dump(0);
            std::cout << ".";
        }
        std::cout << member_ << ", args: [";
        for (const auto& arg : arguments_) {
            arg->dump(0);
            std::cout << ", ";
        }
        std::cout << "])";
    }

    Kind getKind() const override { return Kind::Call; }

    // Add these getter methods
    const ExprAST* getObject() const { return object_.get(); }
    const std::string& getMemberName() const { return member_; }
    const std::vector<std::unique_ptr<ExprAST>>& getArguments() const { return arguments_; }
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
    std::unique_ptr<ExprAST> value_;
public:
    PrintExprAST(std::unique_ptr<ExprAST> value)
        : value_(std::move(value)) {}
        
    void dump(int indent = 0) const override {
        printIndent(indent);
        std::cout << "print(";
        value_->dump(0);
        std::cout << ")";
    }
    
    const ExprAST* getValue() const { return value_.get(); }
    
    Kind getKind() const override { return Kind::Print; }
};

} // namespace matrix

#endif // MATRIX_AST_H
