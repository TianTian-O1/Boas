#include "frontend/Parser.h"
#include "mlirops/MLIRGen.h"
#include <iostream>
#include <fstream>
#include <sstream>

int main() {
    // 读取输入文件
    std::ifstream file("test/matmul.bs");
    if (!file.is_open()) {
        std::cerr << "Error: Could not open matmul.bs" << std::endl;
        return 1;
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string input = buffer.str();

    // 解析输入生成 AST
    matrix::Lexer lexer(input);
    matrix::Parser parser(lexer);
    
    try {
        auto ast = parser.parse();
        std::cout << "Successfully parsed input file\n";
        
        // 生成 MLIR
        matrix::MLIRGen mlirGen;
        std::string mlir = mlirGen.generateMLIR(ast);
        
        // 打印生成的 MLIR
        std::cout << "Generated MLIR:\n" << mlir << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
