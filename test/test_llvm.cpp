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

bool compileMLIR(const std::string& mlir, const std::string& workDir, const std::string& outputPath, const std::string& mode) {
    // 处理工具路径
    std::string mlirOptPath = stripQuotes(MLIR_OPT_PATH);
    std::string mlirTranslatePath = stripQuotes(MLIR_TRANSLATE_PATH);
    std::string llcPath = stripQuotes(LLC_PATH);
    std::string clangPath = stripQuotes(CLANG_PATH);
    std::string llvmInstallPath = stripQuotes(LLVM_INSTALL_PATH);
    
    // 获取项目根目录
    std::string projectRoot = workDir;
    if (workDir.length() >= 6 && workDir.substr(workDir.length() - 6) == "/build") {
        projectRoot = workDir.substr(0, workDir.length() - 6);
    }
    
    // 使用工作目录存放临时文件
    std::string tempDir = workDir;
    std::string mlirFilePath = tempDir + "/temp.mlir";
    std::string llvmMlirFilePath = tempDir + "/temp.llvm.mlir";
    std::string llFilePath = tempDir + "/temp.ll";
    std::string sFilePath = tempDir + "/temp.s";
    
    std::cout << "\n=== Starting MLIR optimization pipeline ===\n";
    
    // Write MLIR to temporary file
    {
        std::ofstream outFile(mlirFilePath);
        if (!outFile.is_open()) {
            throw std::runtime_error("Failed to create MLIR file");
        }
        outFile << mlir;
    }
    
    // MLIR optimization pipeline
    std::string optCmd = "\"" + mlirOptPath + "\"" + 
        " " + mlirFilePath +
        " --pass-pipeline=\"builtin.module("
        "convert-linalg-to-loops,"        // linalg 到循环
        "convert-scf-to-cf,"              // scf 到 cf
        "convert-arith-to-llvm,"          // arith 到 llvm
        "func.func(canonicalize),"         // 函数级优化
        "convert-func-to-llvm,"           // func 到 llvm
        "finalize-memref-to-llvm,"        // memref 最终转换
        "reconcile-unrealized-casts)\"" +  // 处理未实现的转换
        " -o " + llvmMlirFilePath;
            
    std::cout << "\nExecuting MLIR optimization:\n" << optCmd << "\n";
    if (system(optCmd.c_str()) != 0) {
        // 如果失败，打印输入文件内容以帮助调试
        std::cout << "\nInput MLIR content:\n";
        std::ifstream inFile(mlirFilePath);
        if (inFile.is_open()) {
            std::cout << inFile.rdbuf();
        }
        throw std::runtime_error("Failed to execute mlir-opt");
    }
    
    // 检查转换后的文件
    std::cout << "\nChecking converted MLIR:\n";
    std::ifstream convertedFile(llvmMlirFilePath);
    if (convertedFile.is_open()) {
        std::cout << convertedFile.rdbuf() << "\n";
    }
    
    // Convert to LLVM IR
    std::string translateCmd = "\"" + mlirTranslatePath + "\"" +
        " --mlir-to-llvmir" +
        " " + llvmMlirFilePath +
        " -o " + llFilePath;
            
    std::cout << "\nConverting to LLVM IR:\n" << translateCmd << "\n";
    if (system(translateCmd.c_str()) != 0) {
        throw std::runtime_error("Failed to execute mlir-translate");
    }
    
    // Compile to assembly with optimization flags
    std::string llcCmd = "\"" + llcPath + "\"" + 
        " " + llFilePath +
        " -O3" +                           // 最高优化级别
        " -mcpu=native" +                  // 使用本机 CPU 特性
        " -mattr=+avx2,+fma" +            // 启用 AVX2 和 FMA 指令
        " -enable-unsafe-fp-math" +        // 启用不安全的浮点数学优化
        " -fp-contract=fast" +             // 快速浮点数合约
        " -o " + sFilePath;
    
    std::cout << "\nGenerating optimized assembly:\n" << llcCmd << "\n";
    if (system(llcCmd.c_str()) != 0) {
        throw std::runtime_error("Failed to execute llc");
    }
    
    // Final compilation with clang
    std::string clangCmd = "\"" + clangPath + "\"" + 
        " " + sFilePath + 
        " -O3" +                           
        " -march=native" +                 
        " -ffast-math" +                   
        " -fno-math-errno" +              
        " -fno-trapping-math" +           
        " -ffinite-math-only" +           
        " -mllvm -force-vector-width=8" +  
        " -mllvm -force-vector-interleave=4" +  
        " -L" + llvmInstallPath + "/lib" +
        " -L" + projectRoot + "/build/lib" +
        " -Wl,-rpath," + llvmInstallPath + "/lib" +
        " -Wl,-rpath," + projectRoot + "/build/lib" +
        " -lmatrix-runtime" +
        " -lmlir_runner_utils" +
        " -lmlir_c_runner_utils" +
        " -lmlir_float16_utils" +
        " -o " + outputPath;
        
    std::cout << "\nCompiling to executable with optimizations:\n" << clangCmd << "\n";
    if (system(clangCmd.c_str()) != 0) {
        throw std::runtime_error("Failed to execute clang");
    }

    // 清理临时文件
    std::remove(mlirFilePath.c_str());
    std::remove(llvmMlirFilePath.c_str());
    std::remove(llFilePath.c_str());
    std::remove(sFilePath.c_str());
    
    std::cout << "\n=== MLIR compilation pipeline completed successfully ===\n";
    
    // 如果是运行模式，设置库路径并执行
    if (mode == "--run") {
        std::cout << "\nExecuting generated program...\n";
        
        // 设置库路径
        std::string currentPath = getenv("DYLD_LIBRARY_PATH") ? getenv("DYLD_LIBRARY_PATH") : "";
        std::string newPath = llvmInstallPath + "/lib:" + currentPath;
        setenv("DYLD_LIBRARY_PATH", newPath.c_str(), 1);
        
        std::string execCmd;
        if (outputPath.length() > 0 && outputPath[0] == '/') {
            execCmd = outputPath;
        } else {
            execCmd = "./" + outputPath;
        }
        if (system(execCmd.c_str()) != 0) {
            throw std::runtime_error("Failed to execute generated program");
        }
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
        
        // 编译MLIR代码
        std::string workDir = std::filesystem::current_path().string();
        if (!compileMLIR(mlir, workDir, outputFile, mode)) {  // 使用 compileMLIR 而不是 compileToExecutable
            throw std::runtime_error("Compilation failed");
        }
        
        std::cout << "Successfully compiled to: " << outputFile << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
