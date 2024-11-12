#include "frontend/Parser.h"
#include <iostream>
#include <fstream>
#include <sstream>

int main() {
    // 读取 matmul.bs 文件
    std::ifstream file("test/matmul.bs");
    if (!file.is_open()) {
        std::cerr << "Error: Could not open matmul.bs" << std::endl;
        return 1;
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string input = buffer.str();

    std::cout << "=== Parsing Input ===" << std::endl;
    std::cout << input << std::endl;

    // 创建词法分析器和语法分析器
    matrix::Lexer lexer(input);
    matrix::Parser parser(lexer);
    
    try {
        // 解析输入生成 AST
        auto ast = parser.parse();
        
        std::cout << "\n=== Generated AST ===" << std::endl;
        
        // 打印生成的 AST
        std::cout << "Program AST:\n";
        for (const auto& node : ast) {
            node->dump();
            std::cout << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}