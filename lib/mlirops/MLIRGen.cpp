#include "mlirops/MLIRGen.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Region.h"

namespace matrix {


// Add this implementation to MLIRGen.cpp

// In MLIRGen.cpp
MLIRGen::MLIRGen() {
    llvm::errs() << "Initializing MLIRGen...\n";
    context = std::make_unique<mlir::MLIRContext>();
    if (!context) {
        llvm::errs() << "Failed to create MLIRContext\n";
        return;
    }
    
    // Load and register all dialects
    context->loadDialect<mlir::func::FuncDialect>();
    context->loadDialect<mlir::arith::ArithDialect>();
    context->loadDialect<mlir::memref::MemRefDialect>();
    context->loadDialect<mlir::linalg::LinalgDialect>();
    context->loadDialect<mlir::tensor::TensorDialect>();
    
    // Create builder
    builder = std::make_unique<mlir::OpBuilder>(context.get());
    if (!builder) {
        llvm::errs() << "Failed to create OpBuilder\n";
        return;
    }
    
    llvm::errs() << "MLIRGen initialized successfully\n";
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

// Update the generateMLIRForNode function in MLIRGen.cpp

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
            
            default:
                std::cerr << "Error: Unhandled node kind: " << node->getKind() << "\n";
                return nullptr;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error in generateMLIRForNode: " << e.what() << "\n";
        return nullptr;
    }
}

// Updated Tensor handling
mlir::Value MLIRGen::generateMLIRForTensor(const TensorExprAST* expr) {
    std::cerr << "[DEBUG] Generating Tensor operation\n";
    
    // TensorExprAST contains a vector of elements
    const auto& elements = expr->getElements();
    if (elements.empty()) {
        std::cerr << "Error: Empty tensor elements\n";
        return nullptr;
    }
    
    // Process elements
    std::vector<mlir::Value> values;
    for (const auto& element : elements) {
        auto value = generateMLIRForNode(element.get());
        if (!value) {
            std::cerr << "Error: Failed to generate tensor element\n";
            return nullptr;
        }
        values.push_back(value);
    }
    
    // Create tensor from values
    // For now, we'll assume it's a 1D tensor and we'll need to reshape it
    auto elementType = builder->getF64Type();
    auto shape = mlir::MemRefType::get({static_cast<int64_t>(values.size())}, elementType);
    
    auto result = builder->create<mlir::memref::AllocOp>(
        builder->getUnknownLoc(),
        shape
    );
    
    // Initialize the tensor with values
    for (size_t i = 0; i < values.size(); ++i) {
        auto idx = builder->create<mlir::arith::ConstantIndexOp>(
            builder->getUnknownLoc(),
            i
        );
        
        builder->create<mlir::memref::StoreOp>(
            builder->getUnknownLoc(),
            values[i],
            result,
            mlir::ValueRange{idx}
        );
    }
    
    return result;
}


// Add new method for handling Call expressions
mlir::Value MLIRGen::generateMLIRForCall(const CallExprAST* expr) {
    std::cerr << "[DEBUG] Generating Call operation\n";
    // Get the function name
    const std::string& callee = expr->getCallee();
    
    // Handle specific function calls
    if (callee == "matmul") {
        // Handle matrix multiplication
        if (expr->getArgs().size() != 2) {
            std::cerr << "Error: matmul requires exactly 2 arguments\n";
            return nullptr;
        }
        auto lhs = generateMLIRForNode(expr->getArgs()[0].get());
        auto rhs = generateMLIRForNode(expr->getArgs()[1].get());
        if (!lhs || !rhs) return nullptr;
        
        // Create matmul operation
        // ... (rest of matmul implementation)
    }
    
    std::cerr << "Error: Unknown function call: " << callee << "\n";
    return nullptr;
}

// Add new method for handling Array expressions
mlir::Value MLIRGen::generateMLIRForArray(const ArrayExprAST* expr) {
    std::cerr << "[DEBUG] Generating Array operation\n";
    // Implementation for array operations
    return nullptr;
}


mlir::Value MLIRGen::generateMLIRForFunction(const FunctionAST* func) {
    if (!func) {
        std::cerr << "Error: Null function AST\n";
        return nullptr;
    }

    std::cerr << "[DEBUG] Generating MLIR for function: " << func->getName() << "\n";

    try {
        // Process function body
        const auto& body = func->getBody();
        std::cerr << "[DEBUG] Processing function body with " << body.size() << " statements\n";

        mlir::Value lastValue;
        for (const auto& stmt : body) {
            if (!stmt) {
                std::cerr << "Warning: Null statement in function body\n";
                continue;
            }

            std::cerr << "[DEBUG] Processing statement kind: " << stmt->getKind() << "\n";
            lastValue = generateMLIRForNode(stmt.get());
            
            if (!lastValue) {
                std::cerr << "Warning: Failed to generate MLIR for statement\n";
            }
        }

        return lastValue;
    } catch (const std::exception& e) {
        std::cerr << "Error in generateMLIRForFunction: " << e.what() << "\n";
        return nullptr;
    }
}

// Update the generateMLIR function in MLIRGen.cpp

mlir::ModuleOp MLIRGen::generateMLIR(const std::vector<std::unique_ptr<ExprAST>>& ast) {
    std::cerr << "=== Starting MLIR Generation ===\n";
    std::cerr << "Number of AST nodes: " << ast.size() << "\n";
    
    try {
        // Create module
        module = mlir::ModuleOp::create(builder->getUnknownLoc());
        if (!module) {
            std::cerr << "Failed to create ModuleOp\n";
            return nullptr;
        }
        std::cerr << "Created MLIR module successfully\n";
        
        // Set insertion point
        builder->setInsertionPointToStart(module.getBody());
        
        // First, declare the printMemrefF64 function
        auto printMemrefType = mlir::FunctionType::get(
            context.get(),
            {mlir::UnrankedMemRefType::get(builder->getF64Type(), 0)},
            {}
        );
        
        auto printFunc = mlir::func::FuncOp::create(
            builder->getUnknownLoc(),
            "printMemrefF64",
            printMemrefType
        );
        
        if (!printFunc) {
            std::cerr << "Failed to create print function declaration\n";
            return nullptr;
        }
        
        // Mark the function as external
        printFunc.setPrivate();
        printFunc->setAttr("llvm.emit_c_interface", builder->getUnitAttr());
        
        // Add the function declaration to the module
        module.push_back(printFunc);
        std::cerr << "Added printMemrefF64 function declaration\n";
        
        // Create main function
        auto mainType = mlir::FunctionType::get(context.get(), {}, {builder->getI32Type()});
        auto mainFunc = mlir::func::FuncOp::create(
            builder->getUnknownLoc(),
            "main",
            mainType
        );
        
        if (!mainFunc) {
            std::cerr << "Failed to create main function\n";
            return nullptr;
        }
        
        auto* entryBlock = mainFunc.addEntryBlock();
        if (!entryBlock) {
            std::cerr << "Failed to create entry block\n";
            return nullptr;
        }
        
        builder->setInsertionPointToStart(entryBlock);
        std::cerr << "Created main function with entry block\n";
        
        // Process AST nodes
        for (const auto& node : ast) {
            if (!node) {
                std::cerr << "Warning: Null AST node encountered\n";
                continue;
            }
            
            std::cerr << "\nProcessing node kind: " << node->getKind() << "\n";
            
            // Set insertion point before processing each node
            builder->setInsertionPointToEnd(entryBlock);
            
            try {
                mlir::Value value = generateMLIRForNode(node.get());
                if (!value) {
                    std::cerr << "Warning: Node processing returned null value\n";
                }
            } catch (const std::exception& e) {
                std::cerr << "Error processing node: " << e.what() << "\n";
                return nullptr;
            }
        }
        
        // Add return statement
        builder->setInsertionPointToEnd(entryBlock);
        auto returnValue = builder->create<mlir::arith::ConstantIntOp>(
            builder->getUnknownLoc(),
            0,
            32
        );
        
        builder->create<mlir::func::ReturnOp>(
            builder->getUnknownLoc(),
            mlir::ValueRange{returnValue}
        );
        
        module.push_back(mainFunc);
        
        if (failed(mlir::verify(module))) {
            std::cerr << "Module verification failed\n";
            module.dump();
            return nullptr;
        }
        
        std::cerr << "MLIR generation completed successfully\n";
        return module;
        
    } catch (const std::exception& e) {
        std::cerr << "Error in generateMLIR: " << e.what() << "\n";
        return nullptr;
    }
}



mlir::Value MLIRGen::generateMLIRForImport(const ImportAST* import) {
    if (!import) {
        std::cerr << "Error: Null import AST\n";
        return nullptr;
    }

    std::cerr << "[DEBUG] Processing import: " << import->getModuleName() << "\n";
    
    // For now, we just register that we've seen the import
    // No actual MLIR generation needed for import statements
    return builder->create<mlir::arith::ConstantIntOp>(
        builder->getUnknownLoc(),
        0,
        32
    );
}

mlir::Value MLIRGen::generateMLIRForAssignment(const AssignmentExprAST* expr) {
    auto value = generateMLIRForNode(expr->getValue());
    if (!value) return nullptr;
    
    symbolTable[expr->getName()] = value;
    return value;
}

// 其他代码保持不变...
mlir::FloatType MLIRGen::getF64Type() {
    return builder->getF64Type();
}

mlir::Value MLIRGen::createConstantF64(double value) {
    return builder->create<mlir::arith::ConstantFloatOp>(
        builder->getUnknownLoc(),
        llvm::APFloat(value),
        getF64Type());
}

mlir::Value MLIRGen::generateNumberMLIR(const NumberExprAST* number) {
    return createConstantF64(number->getValue());
}

// Helper methods
mlir::MemRefType MLIRGen::getMemRefType(int64_t rows, int64_t cols) {
    return mlir::MemRefType::get({rows, cols}, getF64Type());
}

mlir::Value MLIRGen::createConstantIndex(int64_t value) {
    return builder->create<mlir::arith::ConstantIndexOp>(
        builder->getUnknownLoc(), 
        value
    );
}

void MLIRGen::processASTNode(mlir::Block* block, const std::vector<std::unique_ptr<ExprAST>>& ast) {
    builder->setInsertionPoint(block, block->begin());
    
    for (const auto& node : ast) {
        llvm::errs() << "[DEBUG] Processing AST node kind: " << node->getKind() << "\n";
        
        if (const auto* funcAST = dynamic_cast<const FunctionAST*>(node.get())) {
            llvm::errs() << "[DEBUG] Found function: " << funcAST->getName() << "\n";
            for (const auto& bodyNode : funcAST->getBody()) {
                llvm::errs() << "[DEBUG] Processing function body node kind: " << bodyNode->getKind() << "\n";
                if (auto value = generateMLIRForNode(bodyNode.get())) {
                    if (const auto* assign = dynamic_cast<const AssignmentExprAST*>(bodyNode.get())) {
                        symbolTable[assign->getName()] = value;
                        llvm::errs() << "[DEBUG] Stored " << assign->getName() << " in symbol table\n";
                    }
                } else {
                    llvm::errs() << "[DEBUG] Failed to generate MLIR for body node\n";
                }
            }
        } else {
            if (generateMLIRForNode(node.get())) {
                llvm::errs() << "[DEBUG] Generated MLIR for top-level node\n";
            } else {
                llvm::errs() << "[DEBUG] Failed to generate MLIR for top-level node\n";
            }
        }
    }
}

mlir::Value MLIRGen::generateMLIRForTensorCreate(const TensorCreateExprAST* expr) {
    llvm::errs() << "Creating tensor...\n";
    
    auto rowsNum = static_cast<const NumberExprAST*>(expr->getRows())->getValue();
    auto colsNum = static_cast<const NumberExprAST*>(expr->getCols())->getValue();
    
    auto resultType = mlir::MemRefType::get(
        {static_cast<int64_t>(rowsNum), static_cast<int64_t>(colsNum)},
        builder->getF64Type()
    );
    
    auto result = builder->create<mlir::memref::AllocOp>(
        builder->getUnknownLoc(),
        resultType
    );
    
    const auto& values = expr->getValues();
    int idx = 0;
    
    for (int i = 0; i < rowsNum; i++) {
        for (int j = 0; j < colsNum; j++) {
            auto val = builder->create<mlir::arith::ConstantOp>(
                builder->getUnknownLoc(),
                builder->getF64FloatAttr(
                    static_cast<const NumberExprAST*>(values[idx++].get())->getValue()
                )
            );
            
            auto iVal = createConstantIndex(i);
            auto jVal = createConstantIndex(j);
            
            builder->create<mlir::memref::StoreOp>(
                builder->getUnknownLoc(),
                val,
                result,
                mlir::ValueRange{iVal, jVal}
            );
        }
    }
    
    llvm::errs() << "Tensor created successfully\n";
    return result;
}

std::string MLIRGen::getMLIRString(mlir::ModuleOp module) {
    std::string output;
    llvm::raw_string_ostream os(output);
    module.print(os);
    return output;
}

mlir::Value MLIRGen::generateMLIRForVariable(const VariableExprAST* expr) {
    auto it = symbolTable.find(expr->getName());
    if (it != symbolTable.end()) {
        return it->second;
    }
    
    llvm::errs() << "[DEBUG] Variable " << expr->getName() << " not found in symbol table\n";
    return nullptr;
}

mlir::Value MLIRGen::generateMLIRForBinary(const BinaryExprAST* expr) {
    llvm::errs() << "[DEBUG] Processing Binary node\n";
    llvm::errs() << "[DEBUG] Binary operator: '" << expr->getOp() << "'\n";
    
    if (expr->getOp() == "matmul") {
        llvm::errs() << "[DEBUG] Found matmul operation\n";
        
        // 获取左右操作数
        auto* lhs = static_cast<const VariableExprAST*>(expr->getLHS());
        auto* rhs = static_cast<const VariableExprAST*>(expr->getRHS());
        
        llvm::errs() << "[DEBUG] Matmul operands: " << lhs->getName() << " * " << rhs->getName() << "\n";
        
        // 从符号表中获取变量
        auto lhsValue = symbolTable[lhs->getName()];
        auto rhsValue = symbolTable[rhs->getName()];
        
        if (!lhsValue || !rhsValue) {
            llvm::errs() << "[DEBUG] Failed to find operands in symbol table\n";
            return nullptr;
        }
        
        // 创建 MatmulExprAST 对象
        auto matmulExpr = std::make_unique<MatmulExprAST>(
            std::make_unique<VariableExprAST>(lhs->getName()),
            std::make_unique<VariableExprAST>(rhs->getName())
        );
        
        // 生成矩阵乘法操作
        auto result = generateMLIRForMatmul(matmulExpr.get());
        
        // 检查结果类型
        if (!result || !mlir::isa<mlir::MemRefType>(result.getType())) {
            llvm::errs() << "[DEBUG] Matmul result must be memref type\n";
            return nullptr;
        }
        
        // 如果这是一个赋值操作的一部分，结果会被上层处理
        return result;
    }
    
    return nullptr;
}

// Helper function to validate matrix dimensions
bool validateMatrixDimensions(int64_t rows, int64_t cols, const char* matrixName) {
    if (rows <= 0 || cols <= 0) {
        std::cerr << "Invalid dimensions for " << matrixName << ": " << rows << "x" << cols << "\n";
        return false;
    }
    if (rows > 2048 || cols > 2048) {  // Add reasonable limits
        std::cerr << "Matrix dimensions too large for " << matrixName << ": " << rows << "x" << cols << "\n";
        return false;
    }
    return true;
}

mlir::Value MLIRGen::generateMLIRForTensorRandom(const TensorRandomExprAST* expr) {
    std::cerr << "[DEBUG] Generating random tensor\n";
    auto* rowsNum = static_cast<const NumberExprAST*>(expr->getRows());
    auto* colsNum = static_cast<const NumberExprAST*>(expr->getCols());
    
    int64_t rows = static_cast<int64_t>(rowsNum->getValue());
    int64_t cols = static_cast<int64_t>(colsNum->getValue());
    
    if (rows <= 0 || cols <= 0 || rows > 2048 || cols > 2048) {
        std::cerr << "Invalid matrix dimensions: " << rows << "x" << cols << "\n";
        return nullptr;
    }
    
    auto resultType = mlir::MemRefType::get({rows, cols}, builder->getF64Type());
    
    // Allocate memory
    auto alloc = builder->create<mlir::memref::AllocOp>(
        builder->getUnknownLoc(),
        resultType
    );
    
    // Use a simple deterministic method for random values
    unsigned int seed = rows * cols + rows + cols;
    srand(seed);
    
    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 0; j < cols; ++j) {
            // Generate random value between 0 and 1
            double randVal = static_cast<double>(rand()) / RAND_MAX;
            
            auto val = builder->create<mlir::arith::ConstantFloatOp>(
                builder->getUnknownLoc(),
                llvm::APFloat(randVal),
                builder->getF64Type()
            );
            
            auto iVal = builder->create<mlir::arith::ConstantIndexOp>(
                builder->getUnknownLoc(),
                i
            );
            
            auto jVal = builder->create<mlir::arith::ConstantIndexOp>(
                builder->getUnknownLoc(),
                j
            );
            
            builder->create<mlir::memref::StoreOp>(
                builder->getUnknownLoc(),
                val,
                alloc,
                mlir::ValueRange{iVal, jVal}
            );
        }
    }
    
    return alloc;
}

mlir::Value MLIRGen::generateMLIRForMatmul(const MatmulExprAST* expr) {
    std::cerr << "[DEBUG] Generating matrix multiplication\n";
    
    auto lhs = generateMLIRForNode(expr->getLHS());
    auto rhs = generateMLIRForNode(expr->getRHS());
    
    if (!lhs || !rhs) {
        std::cerr << "Failed to get operands for matmul\n";
        return nullptr;
    }
    
    auto lhsType = mlir::dyn_cast<mlir::MemRefType>(lhs.getType());
    auto rhsType = mlir::dyn_cast<mlir::MemRefType>(rhs.getType());
    
    if (!lhsType || !rhsType) {
        std::cerr << "Invalid operand types for matmul\n";
        return nullptr;
    }
    
    int64_t M = lhsType.getShape()[0];
    int64_t K = lhsType.getShape()[1];
    int64_t N = rhsType.getShape()[1];
    
    if (K != rhsType.getShape()[0]) {
        std::cerr << "Incompatible matrix dimensions for multiplication: "
                  << M << "x" << K << " * " << rhsType.getShape()[0] << "x" << N << "\n";
        return nullptr;
    }
    
    if (M <= 0 || N <= 0 || M > 2048 || N > 2048) {
        std::cerr << "Invalid result matrix dimensions: " << M << "x" << N << "\n";
        return nullptr;
    }
    
    auto resultType = mlir::MemRefType::get({M, N}, builder->getF64Type());
    
    // Allocate result memory
    auto result = builder->create<mlir::memref::AllocOp>(
        builder->getUnknownLoc(),
        resultType
    );
    
    // Initialize result to 0
    auto zero = builder->create<mlir::arith::ConstantFloatOp>(
        builder->getUnknownLoc(),
        llvm::APFloat(0.0),
        builder->getF64Type()
    );
    
    builder->create<mlir::linalg::FillOp>(
        builder->getUnknownLoc(),
        mlir::ValueRange{zero},
        mlir::ValueRange{result}
    );
    
    // Create the matmul operation
    builder->create<mlir::linalg::MatmulOp>(
        builder->getUnknownLoc(),
        mlir::ValueRange{lhs, rhs},
        mlir::ValueRange{result}
    );
    
    // Clean up temporary operands
    if (!isStoredInSymbolTable(lhs)) {
        builder->create<mlir::memref::DeallocOp>(
            builder->getUnknownLoc(),
            lhs
        );
    }
    
    if (!isStoredInSymbolTable(rhs)) {
        builder->create<mlir::memref::DeallocOp>(
            builder->getUnknownLoc(),
            rhs
        );
    }
    
    return result;
}

mlir::Value MLIRGen::generateMLIRForPrint(const PrintExprAST* expr) {
    if (!expr) {
        std::cerr << "Null print expression\n";
        return nullptr;
    }
    
    auto value = generateMLIRForNode(expr->getValue());
    if (!value) {
        std::cerr << "Failed to generate value for print\n";
        return nullptr;
    }
    
    auto memrefType = mlir::dyn_cast<mlir::MemRefType>(value.getType());
    if (!memrefType) {
        std::cerr << "Print value is not a memref type\n";
        return nullptr;
    }
    
    // Cast to unranked memref
    auto unrankedType = mlir::UnrankedMemRefType::get(
        memrefType.getElementType(),
        memrefType.getMemorySpace()
    );
    
    auto castedValue = builder->create<mlir::memref::CastOp>(
        builder->getUnknownLoc(),
        unrankedType,
        value
    );
    
    // Create print call
    builder->create<mlir::func::CallOp>(
        builder->getUnknownLoc(),
        "printMemrefF64",
        mlir::TypeRange{},
        mlir::ValueRange{castedValue}
    );
    
    // Clean up if this is a temporary value
    if (!isStoredInSymbolTable(value)) {
        builder->create<mlir::memref::DeallocOp>(
            builder->getUnknownLoc(),
            value
        );
    }
    
    return value;
}

bool MLIRGen::isStoredInSymbolTable(mlir::Value value) {
    for (const auto& entry : symbolTable) {
        if (entry.second == value) {
            return true;
        }
    }
    return false;
}


} // namespace matrix
