#include "frontend/Parser.h"
#include "mlirops/MLIRGen.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <filesystem>

#ifndef MLIR_OPT_PATH
#error "MLIR_OPT_PATH is not defined"
#endif

#ifndef MLIR_TRANSLATE_PATH
#error "MLIR_TRANSLATE_PATH is not defined"
#endif

#ifndef LLC_PATH
#error "LLC_PATH is not defined"
#endif

#ifndef CLANG_PATH
#error "CLANG_PATH is not defined"
#endif

#ifndef LLVM_INSTALL_PATH
#error "LLVM_INSTALL_PATH is not defined"
#endif

void printUsage(const char* programName) {
    std::cerr << "Usage: " << programName << " [--run | --build] <input_file> [output_file]" << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  --run         : Compile and run the program" << std::endl;
    std::cerr << "  --build       : Compile to binary only" << std::endl;
    std::cerr << "  input_file    : Source file to compile" << std::endl;
    std::cerr << "  output_file   : Output binary file (optional, default: a.out)" << std::endl;
}

bool compileMLIR(const std::string& mlir, const std::string& workDir, const std::string& outputPath) {
    // 写入 MLIR 到临时文件
    std::string mlirPath = workDir + "/temp.mlir";
    {
        std::ofstream mlirFile(mlirPath);
        if (!mlirFile.is_open()) {
            throw std::runtime_error("Failed to create MLIR file");
        }
        mlirFile << mlir;
    }
    std::cout << "Generated MLIR file at: " << mlirPath << std::endl;
    
    // MLIR 优化和转换命令
    std::string llvmMlirPath = workDir + "/temp.llvm.mlir";
    std::string optCmd = std::string("\"") + MLIR_OPT_PATH + "\"" + 
        " " + mlirPath +
        " --convert-linalg-to-loops" +
        " --convert-scf-to-cf" +
        " --convert-cf-to-llvm" +
        " --convert-func-to-llvm" +
        " --convert-arith-to-llvm" +
        " --finalize-memref-to-llvm" +
        " --reconcile-unrealized-casts" +
        " -o " + llvmMlirPath;
            
    std::cout << "Executing command: " << optCmd << std::endl;
    if (system(optCmd.c_str()) != 0) {
        throw std::runtime_error("Failed to execute mlir-opt");
    }

    // 转换为 LLVM IR
    std::string llPath = workDir + "/temp.ll";
    std::string translateCmd = std::string("\"") + MLIR_TRANSLATE_PATH + "\"" +
        " --mlir-to-llvmir " + llvmMlirPath +
        " -o " + llPath;
            
    std::cout << "Executing command: " << translateCmd << std::endl;
    if (system(translateCmd.c_str()) != 0) {
        throw std::runtime_error("Failed to execute mlir-translate");
    }

    // 编译为汇编代码
    std::string sPath = workDir + "/temp.s";
    std::string llcCmd = std::string("\"") + LLC_PATH + "\"" + 
        " " + llPath + " -o " + sPath;
    std::cout << "Executing command: " << llcCmd << std::endl;
    if (system(llcCmd.c_str()) != 0) {
        throw std::runtime_error("Failed to execute llc");
    }

    // 编译为可执行文件
    std::string clangCmd = std::string("\"") + CLANG_PATH + "\"" + 
        " " + sPath + 
        " -L" + LLVM_INSTALL_PATH + "/lib" +
        " -Wl,-rpath," + LLVM_INSTALL_PATH + "/lib" +
        " -lmlir_runner_utils" +
        " -lmlir_c_runner_utils" +
        " -lmlir_float16_utils" +
        " -o " + outputPath;
        
    std::cout << "Executing command: " << clangCmd << std::endl;
    if (system(clangCmd.c_str()) != 0) {
        throw std::runtime_error("Failed to execute clang");
    }

    return true;
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

    std::string workDir = std::filesystem::current_path().string();
    std::string projectRoot = workDir;
    if (workDir.length() >= 6 && workDir.substr(workDir.length() - 6) == "/build") {
        projectRoot = workDir.substr(0, workDir.length() - 6);
    }

    if (!std::filesystem::exists(inputFile)) {
        inputFile = projectRoot + "/" + inputFile;
        if (!std::filesystem::exists(inputFile)) {
            std::cerr << "Error: Input file " << argv[2] << " does not exist" << std::endl;
            return 1;
        }
    }

    try {
        // 解析输入文件
        std::cout << "Processing file: " << inputFile << std::endl;
        std::ifstream file(inputFile);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open input file");
        }
        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string input = buffer.str();

        // 生成 AST
        matrix::Lexer lexer(input);
        matrix::Parser parser(lexer);
        auto ast = parser.parse();
        
        // AST 转换为 MLIR
        matrix::MLIRGen mlirGen;
        auto moduleOp = mlirGen.generateMLIR(ast);
        if (!moduleOp) {
            throw std::runtime_error("Failed to generate MLIR");
        }
        std::string mlir = mlirGen.getMLIRString(moduleOp);
        std::cout << "Generated MLIR:\n" << mlir << "\n\n";

        // 编译 MLIR
        std::string outputPath = std::filesystem::absolute(outputFile).string();
        if (!compileMLIR(mlir, workDir, outputPath)) {
            throw std::runtime_error("Compilation failed");
        }

        std::cout << "Successfully compiled to: " << outputPath << std::endl;

        // 如果是 run 模式，则执行程序
        if (mode == "--run") {
            setenv("DYLD_LIBRARY_PATH", (std::string(LLVM_INSTALL_PATH) + "/lib").c_str(), 1);
            std::cout << "\nExecuting generated program...\n";
            if (system(outputPath.c_str()) != 0) {
                throw std::runtime_error("Failed to execute generated program");
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}