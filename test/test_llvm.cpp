#include "frontend/Parser.h"
#include "mlirops/MLIRGen.h"
#include "mlirops/MLIRToLLVM.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <unistd.h>  // for getcwd

#ifndef LLC_PATH
#define LLC_PATH "llc"
#endif

#ifndef CLANG_PATH
#define CLANG_PATH "clang"
#endif

int main(int argc, char* argv[]) {
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
        // 获取当前工作目录
        char cwd[1024];
        if (getcwd(cwd, sizeof(cwd)) == nullptr) {
            throw std::runtime_error("Failed to get current working directory");
        }
        std::string workDir(cwd);
        std::cout << "Working directory: " << workDir << std::endl;
        std::cout << "Processing file: " << argv[1] << std::endl;

        // 1. 解析输入生成 AST
        matrix::Lexer lexer(input);
        matrix::Parser parser(lexer);
        auto ast = parser.parse();
        
        // 2. AST 转换为 MLIR
        matrix::MLIRGen mlirGen;
        std::string mlir = mlirGen.generateMLIR(ast);
        std::cout << "Generated MLIR:\n" << mlir << "\n\n";
        
        // 3. MLIR 转换为 LLVM IR
        matrix::MLIRToLLVM mlirToLLVM;
        std::string llvmIR = mlirToLLVM.convertToLLVM(mlir);
        std::cout << "Generated LLVM IR:\n" << llvmIR << "\n\n";
        
        // 4. 将 LLVM IR 写入临时文件
        std::string llPath = workDir + "/temp.ll";
        std::ofstream llFile(llPath);
        llFile << llvmIR;
        llFile.close();
        std::cout << "Generated LLVM IR file at: " << llPath << std::endl;
        
        // 修改文件权限
        if (system(("chmod +x " + llPath).c_str()) != 0) {
            throw std::runtime_error("Failed to set file permissions");
        }
        
        // 5. 使用 llc 将 LLVM IR 编译成汇编代码
        std::string sPath = workDir + "/temp.s";
        std::string llc_cmd = std::string("\"") + LLC_PATH + "\" " + llPath + " -o " + sPath;
        std::cout << "Executing command: " << llc_cmd << std::endl;
        if (system(llc_cmd.c_str()) != 0) {
            throw std::runtime_error("Failed to execute llc");
        }
        
        // 6. 使用 clang 将汇编代码编译成可执行文件
        std::string exePath = workDir + "/temp.exe";
        std::string clang_cmd = std::string(CLANG_PATH) + " " + sPath + " -o " + exePath;
        std::cout << "Executing command: " << clang_cmd << std::endl;
        if (system(clang_cmd.c_str()) != 0) {
            throw std::runtime_error("Failed to execute clang");
        }
        
        // 7. 执行生成的程序
        std::cout << "\nExecuting generated program at: " << exePath << std::endl;
        if (system(exePath.c_str()) != 0) {
            throw std::runtime_error("Failed to execute generated program");
        }
        
        // 暂时注释掉清理代码，以便检查文件
        // system(("rm " + llPath + " " + sPath + " " + exePath).c_str());
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
