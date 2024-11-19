// 修改 test_llvm.cpp 为 main.cpp
#include "frontend/Parser.h"
#include "mlirops/MLIRGen.h"
#include "mlirops/MLIRToC.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <filesystem>

void printUsage(const char* programName) {
    std::cerr << "Usage: " << programName << " [--run | --build] <input_file> [output_file]" << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  --run         : Compile and run the program" << std::endl;
    std::cerr << "  --build       : Compile to binary only" << std::endl;
    std::cerr << "  input_file    : Source file to compile" << std::endl;
    std::cerr << "  output_file   : Output binary file (optional, default: a.out)" << std::endl;
}

bool compileToExecutable(const std::string& cCode, 
                        const std::string& workDir,
                        const std::string& outputPath) {
    // 写入C代码到临时文件
    std::string cPath = workDir + "/temp.c";
    {
        std::ofstream cFile(cPath);
        if (!cFile.is_open()) {
            throw std::runtime_error("Failed to create C file");
        }
        cFile << cCode;
    }
    
    // 使用系统CC编译器编译
    std::string compileCmd = "cc " + cPath + " -o " + outputPath;
    return system(compileCmd.c_str()) == 0;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printUsage(argv[0]);
        return 1;
    }

    std::string mode = argv[1];
    if (mode != "--run" && mode != "--build") {
        printUsage(argv[0]);
        return 1;
    }

    std::string inputFile = argv[2];
    std::string outputFile = (argc > 3) ? argv[3] : "a.out";

    try {
        // 解析输入文件
        std::ifstream file(inputFile);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open input file");
        }
        std::stringstream buffer;
        buffer << file.rdbuf();
        
        // 词法和语法分析
        matrix::Lexer lexer(buffer.str());
        matrix::Parser parser(lexer);
        auto ast = parser.parse();
        
        // 生成MLIR
        matrix::MLIRGen mlirGen;
        auto moduleOp = mlirGen.generateMLIR(ast);
        std::string mlir = mlirGen.getMLIRString(moduleOp);
        
        // 转换为C代码
        std::string cCode = matrix::MLIRToC::convertToC(mlir);
        
        // 编译C代码
        std::string workDir = std::filesystem::current_path().string();
        if (!compileToExecutable(cCode, workDir, outputFile)) {
            throw std::runtime_error("Compilation failed");
        }
        
        std::cout << "Successfully compiled to: " << outputFile << std::endl;
        
        // 如果是run模式，则执行程序
        if (mode == "--run") {
            std::cout << "\nExecuting generated program...\n";
            if (system(("./" + outputFile).c_str()) != 0) {
                throw std::runtime_error("Failed to execute generated program");
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}