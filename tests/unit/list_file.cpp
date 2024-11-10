#include <gtest/gtest.h>
#include <fstream>
#include <sstream>
#include "boas/frontend/python_frontend.h"
#include "boas/backend/mlir/MLIRGen.h"
#include "mlir/IR/MLIRContext.h"
#include "boas/backend/mlir/BoasDialect.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "boas/backend/mlir/BoasToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/Support/TargetSelect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"

int main(int argc, char **argv) {
    // 初始化 LLVM targets
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    
    // 创建方言注册表
    mlir::DialectRegistry registry;
    
    // 注册所有 MLIR 方言
    mlir::registerAllDialects(registry);
    
    // 创建 MLIRContext 并注册所有方言
    mlir::MLIRContext context(registry);
    
    // 注册 LLVM 翻译接口
    mlir::registerLLVMDialectTranslation(context);
    
    // 注册 Boas 方言
    context.loadDialect<boas::mlir::BoasDialect>();
    
    // 注册 LLVM 方言
    context.loadDialect<mlir::LLVM::LLVMDialect>();
    
    // 注册所有到 LLVM IR 的翻译
    mlir::registerAllDialects(registry);
    mlir::registerAsmPrinterCLOptions();
    mlir::registerLLVMDialectTranslation(context);
    
    // 添加必要的初始化
    mlir::registerTransformsPasses();
    mlir::LLVM::registerLLVMPasses();
    
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file>\n";
        return 1;
    }
    
    // 读取输入文件
    std::ifstream input_file(argv[1]);
    if (!input_file) {
        std::cerr << "Error: Cannot open input file " << argv[1] << "\n";
        return 1;
    }
    
    std::stringstream buffer;
    buffer << input_file.rdbuf();
    std::string source = buffer.str();
    
    
    auto parser = std::make_unique<boas::PythonFrontend>();
    auto mlirGen = std::make_unique<boas::mlir::MLIRGen>(context);
    
    // 解析和生成 MLIR
    auto ast = parser->parse(source);
    llvm::outs() << "\n[DEBUG] AST generated\n";
    
    mlirGen->generateModuleOp(ast.get());
    llvm::outs() << "\n[DEBUG] MLIR module generated\n";
    
    auto module = mlirGen->getModule();
    
    // 打印生成的 MLIR
    llvm::outs() << "\n[DEBUG] Generated MLIR:\n";
    module.print(llvm::outs());
    
    // 创建 PassManager
    mlir::PassManager pm(&context);
    
    // 添加转换 pass
    boas::mlir::BoasToLLVMTypeConverter typeConverter(&context);
    mlir::RewritePatternSet patterns(&context);
    boas::mlir::populateBoasToLLVMConversionPatterns(typeConverter, patterns);

    auto convertPass = boas::mlir::createConvertBoasToLLVMPass();
    pm.addPass(std::move(convertPass));
    
    // 运行转换
    if (mlir::failed(pm.run(module))) {
        llvm::errs() << "Failed to convert to LLVM IR\n";
        return 1;
    }
    
    // 验证模块
    if (mlir::failed(module.verify())) {
        llvm::errs() << "Module verification failed after conversion\n";
        return 1;
    }
    
    llvm::outs() << "\n[DEBUG] After conversion to LLVM IR:\n";
    module.print(llvm::outs());
    
    // 创建执行引擎
    mlir::ExecutionEngineOptions engineOptions;
    engineOptions.transformer = mlir::makeOptimizingTransformer(/*optLevel=*/3, /*sizeLevel=*/0, /*targetMachine=*/nullptr);

    auto maybeEngine = mlir::ExecutionEngine::create(module, engineOptions);
    if (!maybeEngine) {
        llvm::errs() << "Failed to create execution engine\n";
        return 1;
    }
    
    auto &engine = maybeEngine.get();
    
    // 运行 main 函数
    auto invocationResult = engine->invokePacked("main");
    if (invocationResult) {
        llvm::errs() << "JIT execution failed\n";
        return 1;
    }
    
    return 0;
}