#pragma once

#include <memory>
#include <string>
#include <vector>
#include <variant>
#include <map>

namespace matrix {

// 前向声明
class Node;
class Expression;
class Statement;

// 值类型
using Value = std::variant<int, double, std::string, bool>;

// AST节点基类
class Node {
public:
    virtual ~Node() = default;
    virtual std::string toString() const = 0;
};

// 表达式基类
class Expression : public Node {
public:
    virtual ~Expression() = default;
};

// 字面量
class Literal : public Expression {
public:
    explicit Literal(Value value) : value_(value) {}
    std::string toString() const override;
    const Value& getValue() const { return value_; }

private:
    Value value_;
};

// 标识符
class Identifier : public Expression {
public:
    explicit Identifier(std::string name) : name_(std::move(name)) {}
    std::string toString() const override { return name_; }
    const std::string& getName() const { return name_; }

private:
    std::string name_;
};

// 二元操作
class BinaryOp : public Expression {
public:
    enum class OpType {
        Add, Sub, Mul, Div, Eq, Lt, Gt
    };

    BinaryOp(std::unique_ptr<Expression> left,
            OpType op,
            std::unique_ptr<Expression> right)
        : left_(std::move(left)), op_(op), right_(std::move(right)) {}

    std::string toString() const override;

private:
    std::unique_ptr<Expression> left_;
    OpType op_;
    std::unique_ptr<Expression> right_;
};

// 函数调用
class Call : public Expression {
public:
    Call(std::string func_name,
         std::vector<std::unique_ptr<Expression>> args)
        : func_name_(std::move(func_name)), args_(std::move(args)) {}

    std::string toString() const override;

private:
    std::string func_name_;
    std::vector<std::unique_ptr<Expression>> args_;
};

// 语句基类
class Statement : public Node {
public:
    virtual ~Statement() = default;
};

// 赋值语句
class Assignment : public Statement {
public:
    Assignment(std::string target,
              std::unique_ptr<Expression> value)
        : target_(std::move(target)), value_(std::move(value)) {}

    std::string toString() const override;

private:
    std::string target_;
    std::unique_ptr<Expression> value_;
};

// 函数定义
class FunctionDef : public Statement {
public:
    FunctionDef(std::string name,
                std::vector<std::string> params,
                std::vector<std::unique_ptr<Statement>> body)
        : name_(std::move(name)), 
          params_(std::move(params)),
          body_(std::move(body)) {}

    std::string toString() const override;

private:
    std::string name_;
    std::vector<std::string> params_;
    std::vector<std::unique_ptr<Statement>> body_;
};

// AST构建器 - 提供类Python语法的接口
class ASTBuilder {
public:
    // 创建字面量
    static std::unique_ptr<Expression> num(int value);
    static std::unique_ptr<Expression> num(double value);
    static std::unique_ptr<Expression> str(const std::string& value);
    static std::unique_ptr<Expression> boolean(bool value);

    // 创建标识符
    static std::unique_ptr<Expression> name(const std::string& name);

    // 创建二元操作
    static std::unique_ptr<Expression> add(std::unique_ptr<Expression> left,
                                         std::unique_ptr<Expression> right);
    static std::unique_ptr<Expression> sub(std::unique_ptr<Expression> left,
                                         std::unique_ptr<Expression> right);
    static std::unique_ptr<Expression> mul(std::unique_ptr<Expression> left,
                                         std::unique_ptr<Expression> right);
    static std::unique_ptr<Expression> div(std::unique_ptr<Expression> left,
                                         std::unique_ptr<Expression> right);

    // 创建函数调用
    static std::unique_ptr<Expression> call(const std::string& func_name,
                                          std::vector<std::unique_ptr<Expression>> args);

    // 创建赋值语句
    static std::unique_ptr<Statement> assign(const std::string& target,
                                           std::unique_ptr<Expression> value);

    // 创建函数定义
    static std::unique_ptr<Statement> def(const std::string& name,
                                        const std::vector<std::string>& params,
                                        std::vector<std::unique_ptr<Statement>> body);
};

} // namespace matrix 