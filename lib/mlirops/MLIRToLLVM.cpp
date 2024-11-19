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

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file>" << std::endl;
        return 1;
    }

    std::string workDir = std::filesystem::current_path().string();
    std::string projectRoot = workDir;
    if (workDir.length() >= 6 && workDir.substr(workDir.length() - 6) == "/build") {
        projectRoot = workDir.substr(0, workDir.length() - 6);
    }

    std::string inputFile = argv[1];
    if (!std::filesystem::exists(inputFile)) {
        inputFile = projectRoot + "/" + argv[1];
        if (!std::filesystem::exists(inputFile)) {
            std::cerr << "Error: Input file " << argv[1] << " does not exist" << std::endl;
            return 1;
        }
    }

    try {
        // 1. Parse input to AST
        std::cout << "Processing file: " << inputFile << std::endl;
        std::ifstream file(inputFile);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open input file");
        }
        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string input = buffer.str();

        matrix::Lexer lexer(input);
        matrix::Parser parser(lexer);
        auto ast = parser.parse();
        
        // 2. AST to MLIR
        matrix::MLIRGen mlirGen;
        auto moduleOp = mlirGen.generateMLIR(ast);
        if (!moduleOp) {
            throw std::runtime_error("Failed to generate MLIR");
        }
        std::string mlir = mlirGen.getMLIRString(moduleOp);
        std::cout << "Generated MLIR:\n" << mlir << "\n\n";
        
        // 3. Write MLIR to temp file
        std::string mlirPath = workDir + "/temp.mlir";
        {
            std::ofstream mlirFile(mlirPath);
            if (!mlirFile.is_open()) {
                throw std::runtime_error("Failed to create MLIR file");
            }
            mlirFile << mlir;
        }
        std::cout << "Generated MLIR file at: " << mlirPath << std::endl;
        
        // 4. Use mlir-opt for optimization and conversion
        std::string llvmMlirPath = workDir + "/temp.llvm.mlir";
        std::string optCmd = std::string("\"") + MLIR_OPT_PATH + "\"" + 
            " " + mlirPath +
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

        // 5. Convert to LLVM IR using mlir-translate
        std::string llPath = workDir + "/temp.ll";
        std::string translateCmd = std::string("\"") + MLIR_TRANSLATE_PATH + "\"" +
            " --mlir-to-llvmir " + llvmMlirPath +
            " -o " + llPath;
            
        std::cout << "Executing command: " << translateCmd << std::endl;
        if (system(translateCmd.c_str()) != 0) {
            throw std::runtime_error("Failed to execute mlir-translate");
        }

        // 6. Compile to assembly
        std::string sPath = workDir + "/temp.s";
        std::string llcCmd = std::string("\"") + LLC_PATH + "\"" + 
            " " + llPath + " -o " + sPath;
        std::cout << "Executing command: " << llcCmd << std::endl;
        if (system(llcCmd.c_str()) != 0) {
            throw std::runtime_error("Failed to execute llc");
        }

        // 7. Compile to executable with runtime libraries
        std::string exePath = workDir + "/temp.exe";
        std::string clangCmd = std::string("\"") + CLANG_PATH + "\"" + 
            " " + sPath + 
            " -L" + LLVM_INSTALL_PATH + "/lib" +
            " -Wl,-rpath," + LLVM_INSTALL_PATH + "/lib" +
            " -lmlir_runner_utils" +
            " -lmlir_c_runner_utils" +
            " -lmlir_float16_utils" +
            " -o " + exePath;
        
        std::cout << "Executing command: " << clangCmd << std::endl;
        if (system(clangCmd.c_str()) != 0) {
            throw std::runtime_error("Failed to execute clang");
        }

        // 8. Set library path and execute
        setenv("DYLD_LIBRARY_PATH", (std::string(LLVM_INSTALL_PATH) + "/lib").c_str(), 1);
        std::cout << "\nExecuting generated program at: " << exePath << std::endl;
        if (system(exePath.c_str()) != 0) {
            throw std::runtime_error("Failed to execute generated program");
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}