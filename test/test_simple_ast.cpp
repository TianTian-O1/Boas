#include "frontend/SimpleAST.h"
#include <iostream>

using namespace matrix;

int main() {
    // 示例：构建一个简单的函数定义
    // 对应的Python代码：
    // def add_multiply(a, b, c):
    //     temp = a + b
    //     result = temp * c
    //     return result

    // 构建函数体
    std::vector<std::unique_ptr<Statement>> body;
    
    // temp = a + b
    body.push_back(
        ASTBuilder::assign("temp",
            ASTBuilder::add(
                ASTBuilder::name("a"),
                ASTBuilder::name("b")
            )
        )
    );
    
    // result = temp * c
    body.push_back(
        ASTBuilder::assign("result",
            ASTBuilder::mul(
                ASTBuilder::name("temp"),
                ASTBuilder::name("c")
            )
        )
    );
    
    // 构建函数定义
    auto func = ASTBuilder::def(
        "add_multiply",
        {"a", "b", "c"},
        std::move(body)
    );
    
    // 打印生成的AST
    std::cout << "Generated AST:\n" << func->toString() << std::endl;
    
    // 示例：构建函数调用
    // result = add_multiply(5, 3, 2)
    std::vector<std::unique_ptr<Expression>> args;
    args.push_back(ASTBuilder::num(5));
    args.push_back(ASTBuilder::num(3));
    args.push_back(ASTBuilder::num(2));
    
    auto call_stmt = ASTBuilder::assign(
        "result",
        ASTBuilder::call("add_multiply", std::move(args))
    );
    
    std::cout << "\nFunction call:\n" << call_stmt->toString() << std::endl;
    
    return 0;
} 