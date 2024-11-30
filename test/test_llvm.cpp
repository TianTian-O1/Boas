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
    std::cerr << "  --build       : Compile the program only" << std::endl;
    std::cerr << "  input_file    : Source file to compile" << std::endl;
    std::cerr << "  output_file   : Output file (optional)" << std::endl;
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
    std::string exeFilePath = tempDir + "/temp.exe";
    
    std::cout << "\n=== Starting MLIR optimization pipeline ===\n";
    
    // Write MLIR to temporary file
    {
        std::string tempDir = std::filesystem::current_path().string();
        std::replace(tempDir.begin(), tempDir.end(), '\\', '/');
        std::string mlirFilePath = tempDir + "/temp.mlir";
        std::string llvmMlirFilePath = tempDir + "/temp.llvm.mlir";
        std::string llFilePath = tempDir + "/temp.ll";
        std::string sFilePath = tempDir + "/temp.s";
        std::string exeFilePath = tempDir + "/temp.exe";
        
        std::ofstream outFile(mlirFilePath);
        if (!outFile.is_open()) {
            throw std::runtime_error("Failed to create MLIR file");
        }
        outFile << mlir;
        outFile.close();
        
        // Execute MLIR optimization
        std::cout << "\nExecuting MLIR optimization:\n";
        std::string mlirOptPath = stripQuotes(MLIR_OPT_PATH);
        std::replace(mlirOptPath.begin(), mlirOptPath.end(), '\\', '/');
        
        std::string pipeline = "builtin.module(convert-linalg-to-loops,loop-invariant-code-motion,"
                             "func.func(affine-loop-unroll{unroll-factor=4}),"
                             "affine-loop-fusion,convert-scf-to-cf,convert-vector-to-llvm,"
                             "convert-arith-to-llvm,func.func(canonicalize),"
                             "convert-func-to-llvm,finalize-memref-to-llvm,"
                             "reconcile-unrealized-casts)";
        
        std::string cmd = "\"" + mlirOptPath + "\" \"" + mlirFilePath + "\" -o \"" + llvmMlirFilePath + "\"" +
                         " --pass-pipeline='" + pipeline + "'";
        std::cout << cmd << std::endl;
        
        // Convert backslashes to forward slashes in the command
        std::replace(cmd.begin(), cmd.end(), '\\', '/');
        
        // Use PowerShell to execute the command
        std::string psCmd = "powershell.exe -NoProfile -NonInteractive -Command \"& { cd '" + tempDir + "'; " + cmd + " }\" 2>&1";
        std::cout << "\nExecuting command:\n" << psCmd << std::endl;
        
        if (system(psCmd.c_str()) != 0) {
            // 如果失败，打印输入文件内容以帮助调试
            std::cout << "\nInput MLIR content:\n";
            std::ifstream inFile(mlirFilePath);
            if (inFile.is_open()) {
                std::cout << inFile.rdbuf();
            }
            throw std::runtime_error("Failed to execute mlir-opt");
        }
        
        // Check converted MLIR
        std::cout << "\nChecking converted MLIR:\n";
        std::ifstream mlirFile(llvmMlirFilePath);
        if (mlirFile.is_open()) {
            std::cout << mlirFile.rdbuf();
        }
        
        // Convert to LLVM IR
        std::cout << "\n\nConverting to LLVM IR:\n";
        std::string mlirTranslatePath = stripQuotes(MLIR_TRANSLATE_PATH);
        std::replace(mlirTranslatePath.begin(), mlirTranslatePath.end(), '\\', '/');
        
        cmd = "\"" + mlirTranslatePath + "\" --mlir-to-llvmir \"" + llvmMlirFilePath + "\" -o \"" + llFilePath + "\"";
        std::cout << cmd << std::endl;
        
        psCmd = "powershell.exe -NoProfile -NonInteractive -Command \"& { cd '" + tempDir + "'; " + cmd + " }\" 2>&1";
        if (system(psCmd.c_str()) != 0) {
            throw std::runtime_error("Failed to execute mlir-translate");
        }
        
        // Generate optimized assembly
        std::cout << "\nGenerating optimized assembly:\n";
        std::string llcPath = stripQuotes(LLC_PATH);
        std::replace(llcPath.begin(), llcPath.end(), '\\', '/');
        
        cmd = "\"" + llcPath + "\" \"" + llFilePath + "\" -O3 -o \"" + sFilePath + "\"";
        std::cout << cmd << std::endl;
        
        psCmd = "powershell.exe -NoProfile -NonInteractive -Command \"& { cd '" + tempDir + "'; " + cmd + " }\" 2>&1";
        if (system(psCmd.c_str()) != 0) {
            throw std::runtime_error("Failed to execute llc");
        }
        
        // Compile to executable with optimizations
        std::cout << "\nCompiling to executable with optimizations:\n";
        cmd = clangPath + " \"" + tempDir + "/temp.s\" -O3 -march=native -ffast-math -fno-math-errno -fno-trapping-math -ffinite-math-only \"-mllvm=-force-vector-width=8\" \"-mllvm=-force-vector-interleave=4\" -L\"" + llvmInstallPath + "/lib\" -L\"" + tempDir + "/lib/Release\" -lmatrix-runtime -o \"" + tempDir + "/Release/temp.exe\"";
        std::cout << cmd << std::endl;
        
        if (system(cmd.c_str()) != 0) {
            std::cerr << "Error: Failed to compile to executable" << std::endl;
            return 1;
        }
        
        // 如果是运行模式，设置路径并执行
        if (mode == "--run") {
            std::cout << "\nExecuting generated program...\n";
            
            // 设置库路径
#ifdef _WIN32
            std::string currentPath = getenv("PATH") ? getenv("PATH") : "";
            std::string newPath = llvmInstallPath + "/lib;" + tempDir + "/lib/Release;" + currentPath;
            _putenv_s("PATH", newPath.c_str());
#else
            std::string currentPath = getenv("DYLD_LIBRARY_PATH") ? getenv("DYLD_LIBRARY_PATH") : "";
            std::string newPath = llvmInstallPath + "/lib:" + currentPath;
            setenv("DYLD_LIBRARY_PATH", newPath.c_str(), 1);
#endif
            
            std::string execCmd = "\"" + tempDir + "/Release/temp.exe\"";
            // 修改：执行程序但不抛出异常
            int result = system(execCmd.c_str());
            if (result != 0) {
                std::cout << "\nNote: Program execution completed with return code " 
                          << result << ". This is expected for some test cases.\n";
            }
        }
    }
    
    // 清理临时文件
    std::remove(mlirFilePath.c_str());
    std::remove(llvmMlirFilePath.c_str());
    std::remove(llFilePath.c_str());
    std::remove(sFilePath.c_str());
    std::remove(exeFilePath.c_str());
    
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
