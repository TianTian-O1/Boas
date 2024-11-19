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

// 去掉字符串两端的引号
std::string stripQuotes(const std::string& str) {
    if (str.length() >= 2 && str.front() == '"' && str.back() == '"') {
        return str.substr(1, str.length() - 2);
    }
    return str;
}

void printUsage(const char* programName) {
    std::cerr << "Usage: " << programName << " [--run | --build] <input_file> [output_file]" << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  --run         : Compile and run the program" << std::endl;
    std::cerr << "  --build       : Compile to binary only" << std::endl;
    std::cerr << "  input_file    : Source file to compile" << std::endl;
    std::cerr << "  output_file   : Output binary file (optional, default: a.out)" << std::endl;
}

bool compileMLIR(const std::string& mlir, const std::string& workDir, const std::string& outputPath) {
    // 处理工具路径
    std::string mlirOptPath = stripQuotes(MLIR_OPT_PATH);
    std::string mlirTranslatePath = stripQuotes(MLIR_TRANSLATE_PATH);
    std::string llcPath = stripQuotes(LLC_PATH);
    std::string clangPath = stripQuotes(CLANG_PATH);
    std::string llvmInstallPath = stripQuotes(LLVM_INSTALL_PATH);
    
    std::cout << "\n=== Starting MLIR compilation pipeline ===\n";
    std::cout << "Using tools:\n";
    std::cout << "  mlir-opt: " << mlirOptPath << "\n";
    std::cout << "  mlir-translate: " << mlirTranslatePath << "\n";
    std::cout << "  llc: " << llcPath << "\n";
    std::cout << "  clang: " << clangPath << "\n";
    std::cout << "  LLVM install path: " << llvmInstallPath << "\n\n";
    
    // Write MLIR to temporary file
    std::string mlirPath = workDir + "/temp.mlir";
    {
        std::ofstream mlirFile(mlirPath);
        if (!mlirFile.is_open()) {
            throw std::runtime_error("Failed to create MLIR file");
        }
        mlirFile << mlir;
        mlirFile.close();  // 确保文件写入并关闭
    }
    std::cout << "Generated MLIR file at: " << mlirPath << "\n";
    
    // MLIR optimization pipeline
    std::string llvmMlirPath = workDir + "/temp.llvm.mlir";
    std::string optCmd = "\"" + mlirOptPath + "\"" + 
        " " + mlirPath +
        " --convert-linalg-to-loops" +         // 转换 linalg 到循环
        " --convert-scf-to-cf" +               // 转换结构化控制流到 CFG
        " --convert-func-to-llvm" +            // 转换函数到 LLVM
        " --finalize-memref-to-llvm" +         // 转换 memref 到 LLVM
        " --convert-arith-to-llvm" +           // 转换算术操作到 LLVM
        " --convert-vector-to-llvm" +          // 转换向量操作到 LLVM
        " --reconcile-unrealized-casts" +      // 处理未实现的转换
        " --convert-index-to-llvm" +           // 添加索引类型转换
        " -o " + llvmMlirPath;
            
    std::cout << "\nExecuting MLIR optimization:\n" << optCmd << "\n";
    int optResult = system(optCmd.c_str());
    if (optResult != 0) {
        // 如果失败，打印输入文件内容以帮助调试
        std::cout << "\nInput MLIR content:\n";
        std::ifstream inFile(mlirPath);
        if (inFile.is_open()) {
            std::cout << inFile.rdbuf();
        }
        throw std::runtime_error("Failed to execute mlir-opt (exit code: " + std::to_string(optResult) + ")");
    }
    std::cout << "MLIR optimization completed\n";

    // 检查转换后的文件
    std::cout << "\nChecking converted MLIR:\n";
    std::ifstream convertedFile(llvmMlirPath);
    if (convertedFile.is_open()) {
        std::cout << convertedFile.rdbuf() << "\n";
    }

    // Convert to LLVM IR
    std::string llPath = workDir + "/temp.ll";
    std::string translateCmd = "\"" + mlirTranslatePath + "\"" +
        " --mlir-to-llvmir" +
        " " + llvmMlirPath +
        " -o " + llPath;
            
    std::cout << "\nConverting to LLVM IR:\n" << translateCmd << "\n";
    int translateResult = system(translateCmd.c_str());
    if (translateResult != 0) {
        throw std::runtime_error("Failed to execute mlir-translate (exit code: " + std::to_string(translateResult) + ")");
    }
    std::cout << "LLVM IR generation completed\n";

    // Compile to assembly
    std::string sPath = workDir + "/temp.s";
    std::string llcCmd = "\"" + llcPath + "\"" + 
        " " + llPath + " -o " + sPath;
    std::cout << "\nGenerating assembly:\n" << llcCmd << "\n";
    int llcResult = system(llcCmd.c_str());
    if (llcResult != 0) {
        throw std::runtime_error("Failed to execute llc (exit code: " + std::to_string(llcResult) + ")");
    }
    std::cout << "Assembly generation completed\n";

    // Compile to executable
    std::string clangCmd = "\"" + clangPath + "\"" + 
        " " + sPath + 
        " -L" + llvmInstallPath + "/lib" +
        " -Wl,-rpath," + llvmInstallPath + "/lib" +
        " -lmlir_runner_utils" +
        " -lmlir_c_runner_utils" +
        " -lmlir_float16_utils" +
        " -o " + outputPath;
        
    std::cout << "\nCompiling to executable:\n" << clangCmd << "\n";
    int clangResult = system(clangCmd.c_str());
    if (clangResult != 0) {
        throw std::runtime_error("Failed to execute clang (exit code: " + std::to_string(clangResult) + ")");
    }
    std::cout << "Compilation to executable completed\n";
    
    std::cout << "\n=== MLIR compilation pipeline completed successfully ===\n";
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

    try {
        // Check the current directory
        std::string workDir = std::filesystem::current_path().string();
        std::string projectRoot = workDir;
        if (workDir.length() >= 6 && workDir.substr(workDir.length() - 6) == "/build") {
            projectRoot = workDir.substr(0, workDir.length() - 6);
        }

        // Process input file
        if (!std::filesystem::exists(inputFile)) {
            inputFile = projectRoot + "/" + inputFile;
            if (!std::filesystem::exists(inputFile)) {
                std::cerr << "Error: Input file " << argv[2] << " does not exist" << std::endl;
                return 1;
            }
        }
        std::cout << "Processing file: " << inputFile << "\n";

        // Parse input file
        std::ifstream file(inputFile);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open input file");
        }
        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string input = buffer.str();

        // Generate AST
        matrix::Lexer lexer(input);
        matrix::Parser parser(lexer);
        auto ast = parser.parse();
        std::cout << "\nAST generation completed\n";
        
        // Convert AST to MLIR
        std::cout << "\nGenerating MLIR...\n";
        matrix::MLIRGen mlirGen;
        auto moduleOp = mlirGen.generateMLIR(ast);
        if (!moduleOp) {
            throw std::runtime_error("Failed to generate MLIR");
        }
        std::string mlir = mlirGen.getMLIRString(moduleOp);
        std::cout << "\nGenerated MLIR:\n" << mlir << "\n";

        // Compile MLIR
        std::string outputPath = std::filesystem::absolute(outputFile).string();
        if (!compileMLIR(mlir, workDir, outputPath)) {
            throw std::runtime_error("Compilation failed");
        }

        std::cout << "\nSuccessfully compiled to: " << outputPath << "\n";

        // Run if in run mode
        if (mode == "--run") {
            setenv("DYLD_LIBRARY_PATH", (std::string(LLVM_INSTALL_PATH) + "/lib").c_str(), 1);
            std::cout << "\nExecuting generated program...\n";
            std::string runCmd = outputPath;
            if (system(runCmd.c_str()) != 0) {
                throw std::runtime_error("Failed to execute generated program");
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}