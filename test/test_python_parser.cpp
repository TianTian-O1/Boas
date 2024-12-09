#include "frontend/python_parser/PythonASTBuilder.h"
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>

// 读取文件内容
std::string readFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return "";
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

void testPythonParser(const std::string& filename) {
    // 读取测试文件
    std::string testCode = readFile(filename);
    if (testCode.empty()) {
        std::cerr << "Error: Empty file or failed to read file" << std::endl;
        return;
    }

    std::cout << "Source code to parse:\n" << testCode << "\n" << std::endl;

    boas::python::PythonASTBuilder builder;
    auto ast = builder.buildFromSource(testCode);
    
    if (!ast) {
        std::cerr << "Error: Failed to parse code from " << filename << std::endl;
        return;
    }

    std::cout << "Successfully parsed " << filename << "!" << std::endl;
    std::cout << "AST structure:" << std::endl;
    ast->dump(0);
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <test_file>" << std::endl;
        return 1;
    }

    std::cout << "Testing Python parser with file: " << argv[1] << std::endl;
    testPythonParser(argv[1]);
    return 0;
} 