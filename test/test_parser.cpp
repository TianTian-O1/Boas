#include "frontend/Parser.h"
#include "Debug.h"
#include <iostream>
#include <fstream>
#include <sstream>

using namespace matrix;

int main(int argc, char* argv[]) {
    #ifdef DEBUG_MODE
        matrix::Debug::setLevel(DEBUG_LEVEL);
    #endif
    
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file>" << std::endl;
        return 1;
    }
    
    // 读取输入文件
    std::ifstream file(argv[1]);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << argv[1] << std::endl;
        return 1;
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string input = buffer.str();
    
    try {
        // 创建词法分析器和语法分析器
        Lexer lexer(input);
        Parser parser(lexer);
        
        // 解析输入生成 AST
        auto ast = parser.parse();
        std::cout << "Successfully parsed input file\n";
        
        // 打印生成的 AST
        std::cout << "\nGenerated AST:\n";
        for (const auto& node : ast) {
            node->dump();
            std::cout << "\n";
        }
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error during parsing: " << e.what() << std::endl;
        return 1;
    }
}