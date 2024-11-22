// MLIRGen.cpp - 核心实现文件
#include "mlirops/MLIRGen.h"
#include "llvm/Support/raw_ostream.h"

namespace matrix {

MLIRGen::MLIRGen() {
    initializeContext();
}

void MLIRGen::initializeContext() {
    llvm::errs() << "Initializing MLIRGen...\n";
    context = std::make_unique<mlir::MLIRContext>();
    if (!context) {
        llvm::errs() << "Failed to create MLIRContext\n";
        return;
    }

    // Load and register all required dialects
    context->loadDialect<mlir::func::FuncDialect>();
    context->loadDialect<mlir::arith::ArithDialect>();
    context->loadDialect<mlir::memref::MemRefDialect>();
    context->loadDialect<mlir::linalg::LinalgDialect>();
    context->loadDialect<mlir::tensor::TensorDialect>();
    context->loadDialect<mlir::scf::SCFDialect>();
    context->loadDialect<mlir::vector::VectorDialect>();
    
    // Create builder
    builder = std::make_unique<mlir::OpBuilder>(context.get());
    if (!builder) {
        llvm::errs() << "Failed to create OpBuilder\n";
        return;
    }
    
    // Create module
    module = mlir::ModuleOp::create(builder->getUnknownLoc());
    
    // Declare runtime functions
    declareRuntimeFunctions();
    
    llvm::errs() << "MLIRGen initialized successfully with all required dialects\n";
}

void MLIRGen::declareRuntimeFunctions() {
    // printFloat function
    auto printFloatType = mlir::FunctionType::get(
        context.get(),
        {builder->getF64Type()},  // 参数类型：double
        {}                        // 返回类型：void
    );
    
    auto printFloatFunc = mlir::func::FuncOp::create(
        builder->getUnknownLoc(),
        "printFloat",
        printFloatType
    );
    printFloatFunc.setPrivate();
    module.push_back(printFloatFunc);
    
    // printMemrefF64 function
    auto memrefType = mlir::UnrankedMemRefType::get(
        builder->getF64Type(),  // f64 类型
        0                       // memory space
    );
    
    auto printMemrefF64Type = mlir::FunctionType::get(
        context.get(),
        {memrefType},  // 参数类型：memref<*xf64>
        {}            // 返回类型：void
    );
    
    auto printMemrefF64Func = mlir::func::FuncOp::create(
        builder->getUnknownLoc(),
        "printMemrefF64",
        printMemrefF64Type
    );
    printMemrefF64Func.setPrivate();
    module.push_back(printMemrefF64Func);
    
    // system_time_msec function
    auto timeType = mlir::FunctionType::get(
        context.get(),
        {},                      // 无参数
        {builder->getF64Type()}  // 返回类型：double
    );
    
    auto timeFunc = mlir::func::FuncOp::create(
        builder->getUnknownLoc(),
        "system_time_msec",
        timeType
    );
    timeFunc.setPrivate();
    module.push_back(timeFunc);

    // generate_random function
    auto randomType = mlir::FunctionType::get(
        context.get(),
        {},                        // 无参数
        {builder->getF64Type()}    // 返回类型：double
    );
    
    auto randomFunc = mlir::func::FuncOp::create(
        builder->getUnknownLoc(),
        "generate_random",
        randomType
    );
    randomFunc.setPrivate();
    module.push_back(randomFunc);
}

mlir::ModuleOp MLIRGen::generateMLIR(const std::vector<std::unique_ptr<ExprAST>>& ast) {
    std::cerr << "=== Starting MLIR Generation ===\n";
    std::cerr << "Number of AST nodes: " << ast.size() << "\n";
    
    try {
        // Create main function
        auto mainType = mlir::FunctionType::get(context.get(), {}, {builder->getI32Type()});
        auto mainFunc = mlir::func::FuncOp::create(
            builder->getUnknownLoc(),
            "main",
            mainType
        );
        
        auto entryBlock = mainFunc.addEntryBlock();
        builder->setInsertionPointToStart(entryBlock);
        
        module.push_back(mainFunc);
        
        // Process AST nodes
        for (const auto& node : ast) {
            if (!generateMLIRForNode(node.get())) {
                std::cerr << "Failed to generate MLIR for node\n";
                return nullptr;
            }
        }
        
        // Add return statement
        auto zero = builder->create<mlir::arith::ConstantIntOp>(
            builder->getUnknownLoc(),
            0,
            builder->getI32Type()
        );
        builder->create<mlir::func::ReturnOp>(
            builder->getUnknownLoc(),
            mlir::ValueRange{zero}
        );

        return module;
        
    } catch (const std::exception& e) {
        std::cerr << "Error in generateMLIR: " << e.what() << "\n";
        return nullptr;
    }
}

mlir::Value MLIRGen::generateMLIRForNode(const ExprAST* node) {
    if (!node) {
        std::cerr << "Error: Null node passed to generateMLIRForNode\n";
        return nullptr;
    }

    std::cerr << "[DEBUG] generateMLIRForNode kind: " << node->getKind() << "\n";

    try {
        switch (node->getKind()) {
            case ExprAST::Number:
                return generateNumberMLIR(static_cast<const NumberExprAST*>(node));
            case ExprAST::Variable:
                return generateMLIRForVariable(static_cast<const VariableExprAST*>(node));
            case ExprAST::Binary:
                return generateMLIRForBinary(static_cast<const BinaryExprAST*>(node));
            case ExprAST::Call:
                return generateMLIRForCall(static_cast<const CallExprAST*>(node));
            case ExprAST::Import:
                return generateMLIRForImport(static_cast<const ImportAST*>(node));
            case ExprAST::Function:
                return generateMLIRForFunction(static_cast<const FunctionAST*>(node));
            case ExprAST::Print:
                return generateMLIRForPrint(static_cast<const PrintExprAST*>(node));
            case ExprAST::Assignment:
                return generateMLIRForAssignment(static_cast<const AssignmentExprAST*>(node));
            case ExprAST::Array:
                return generateMLIRForArray(static_cast<const ArrayExprAST*>(node));
            case ExprAST::Tensor:
                return generateMLIRForTensor(static_cast<const TensorExprAST*>(node));
            case ExprAST::TensorCreate:
                return generateMLIRForTensorCreate(static_cast<const TensorCreateExprAST*>(node));
            case ExprAST::Matmul:
                return generateMLIRForMatmul(static_cast<const MatmulExprAST*>(node));
            case ExprAST::TensorRandom:
                return generateMLIRForTensorRandom(static_cast<const TensorRandomExprAST*>(node));
            case ExprAST::TimeCall:
                if (auto timeCall = static_cast<const TimeCallExprAST*>(node)) {
                    if (timeCall->getFuncName() == "now") {
                        return generateTimeNowMLIR(timeCall);
                    }
                }
                std::cerr << "Unknown time function\n";
                return nullptr;
            default:
                std::cerr << "Error: Unhandled node kind: " << node->getKind() << "\n";
                return nullptr;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error in generateMLIRForNode: " << e.what() << "\n";
        return nullptr;
    }
}

std::string MLIRGen::getMLIRString(mlir::ModuleOp module) {
    std::string output;
    llvm::raw_string_ostream os(output);
    module.print(os);
    return output;
}

} // namespace matrix

// MLIRGenUtils.cpp - 工具函数实现
#include "mlirops/MLIRGen.h"

namespace matrix {

mlir::FloatType MLIRGen::getF64Type() {
    return builder->getF64Type();
}

mlir::Value MLIRGen::createConstantF64(double value) {
    return builder->create<mlir::arith::ConstantFloatOp>(
        builder->getUnknownLoc(),
        llvm::APFloat(value),
        getF64Type()
    );
}

mlir::Value MLIRGen::createConstantIndex(int64_t value) {
    return builder->create<mlir::arith::ConstantIndexOp>(
        builder->getUnknownLoc(),
        value
    );
}

mlir::MemRefType MLIRGen::getMemRefType(int64_t rows, int64_t cols) {
    return mlir::MemRefType::get({rows, cols}, getF64Type());
}

bool MLIRGen::isStoredInSymbolTable(mlir::Value value) {
    for (const auto& entry : symbolTable) {
        if (entry.second == value) {
            return true;
        }
    }
    return false;
}

void MLIRGen::dumpState(const std::string& message) {
    llvm::errs() << "\n=== " << message << " ===\n";
    if (module) {
        llvm::errs() << "Current MLIR Module:\n";
        module.dump();
    } else {
        llvm::errs() << "No module available\n";
    }
    llvm::errs() << "\nSymbol table contents:\n";
    for (const auto& entry : symbolTable) {
        llvm::errs() << "  " << entry.first << ": ";
        if (entry.second) {
            entry.second.getType().dump();
        } else {
            llvm::errs() << "null value";
        }
        llvm::errs() << "\n";
    }
    llvm::errs() << "==================\n\n";
}

} // namespace matrix