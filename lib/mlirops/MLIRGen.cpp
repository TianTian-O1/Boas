#include "mlirops/MLIRGen.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Region.h"

namespace matrix {

MLIRGen::MLIRGen() {
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
    context->loadDialect<mlir::scf::SCFDialect>();         // 添加 SCF dialect
    context->loadDialect<mlir::vector::VectorDialect>();   // 添加 Vector dialect
    
    // Create builder
    builder = std::make_unique<mlir::OpBuilder>(context.get());
    if (!builder) {
        llvm::errs() << "Failed to create OpBuilder\n";
        return;
    }
    
    // 创建模块
    module = mlir::ModuleOp::create(builder->getUnknownLoc());
    
    // 声明运行时函数
    declareRuntimeFunctions();
    
    llvm::errs() << "MLIRGen initialized successfully with all required dialects\n";
}

void MLIRGen::declareRuntimeFunctions() {
    // 声明 printFloat 函数
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
    
    // 声明 printString 函数
    auto stringType = mlir::UnrankedMemRefType::get(
        builder->getIntegerType(8),  // i8 类型
        0                            // memory space
    );
    
    auto printStringType = mlir::FunctionType::get(
        context.get(),
        {stringType},  // 参数类型：memref<*xi8>
        {}            // 返回类型：void
    );
    
    auto printStringFunc = mlir::func::FuncOp::create(
        builder->getUnknownLoc(),
        "printString",
        printStringType
    );
    printStringFunc.setPrivate();
    module.push_back(printStringFunc);
    
    // 声明 system_time_msec 函数
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

    // Add declaration for generate_random function
    auto randomType = mlir::FunctionType::get(
        context.get(),
        {},                         // no input parameters
        {builder->getF64Type()}    // returns double
    );
    
    auto randomFunc = mlir::func::FuncOp::create(
        builder->getUnknownLoc(),
        "generate_random",
        randomType
    );
    randomFunc.setPrivate();
    module.push_back(randomFunc);
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
        std::cerr << "Incompatible matrix dimensions\n";
        return nullptr;
    }

    // 创建一维数组类型的memref
    auto flatType = mlir::MemRefType::get({M * N}, builder->getF64Type());
    auto result = builder->create<mlir::memref::AllocOp>(builder->getUnknownLoc(), flatType);

    // 创建常量
    auto c0 = builder->create<mlir::arith::ConstantIndexOp>(builder->getUnknownLoc(), 0);
    auto c1 = builder->create<mlir::arith::ConstantIndexOp>(builder->getUnknownLoc(), 1);
    auto cM = builder->create<mlir::arith::ConstantIndexOp>(builder->getUnknownLoc(), M);
    auto cK = builder->create<mlir::arith::ConstantIndexOp>(builder->getUnknownLoc(), K);
    auto cN = builder->create<mlir::arith::ConstantIndexOp>(builder->getUnknownLoc(), N);
    
    auto zero = builder->create<mlir::arith::ConstantFloatOp>(
        builder->getUnknownLoc(),
        llvm::APFloat(0.0),
        builder->getF64Type()
    );

    // 主计算循环
    builder->create<mlir::scf::ForOp>(
        builder->getUnknownLoc(), c0, cM, c1,
        mlir::ValueRange{},
        [&](mlir::OpBuilder& i_builder, mlir::Location i_loc, mlir::Value i, mlir::ValueRange) {
            i_builder.create<mlir::scf::ForOp>(
                i_loc, c0, cN, c1,
                mlir::ValueRange{},
                [&](mlir::OpBuilder& j_builder, mlir::Location j_loc, mlir::Value j, mlir::ValueRange) {
                    // 使用scf.for的iter_args进行累加
                    auto inner_loop = j_builder.create<mlir::scf::ForOp>(
                        j_loc, c0, cK, c1,
                        mlir::ValueRange{zero},  // 初始累加值
                        [&](mlir::OpBuilder& k_builder, mlir::Location k_loc, mlir::Value k, mlir::ValueRange k_args) {
                            // 计算A的一维索引：i * K + k
                            auto a_idx_1 = k_builder.create<mlir::arith::MulIOp>(k_loc, i, cK);
                            auto a_idx = k_builder.create<mlir::arith::AddIOp>(k_loc, a_idx_1, k);

                            // 计算B的一维索引：k * N + j
                            auto b_idx_1 = k_builder.create<mlir::arith::MulIOp>(k_loc, k, cN);
                            auto b_idx = k_builder.create<mlir::arith::AddIOp>(k_loc, b_idx_1, j);

                            // 加载元素
                            auto a_val = k_builder.create<mlir::memref::LoadOp>(k_loc, lhs, mlir::ValueRange{i, k});
                            auto b_val = k_builder.create<mlir::memref::LoadOp>(k_loc, rhs, mlir::ValueRange{k, j});

                            // 相乘
                            auto mul = k_builder.create<mlir::arith::MulFOp>(k_loc, a_val, b_val);

                            // 累加
                            auto sum_iter = k_args[0];
                            auto new_sum = k_builder.create<mlir::arith::AddFOp>(k_loc, sum_iter, mul);

                            // yield新的累加值
                            k_builder.create<mlir::scf::YieldOp>(k_loc, mlir::ValueRange{new_sum});
                        }
                    );

                    // 计算C的一维索引：i * N + j
                    auto c_idx_1 = j_builder.create<mlir::arith::MulIOp>(j_loc, i, cN);
                    auto c_idx = j_builder.create<mlir::arith::AddIOp>(j_loc, c_idx_1, j);

                    // 存储最终结果
                    j_builder.create<mlir::memref::StoreOp>(
                        j_loc,
                        inner_loop.getResult(0),  // 获取累加的最终结果
                        result,
                        mlir::ValueRange{c_idx}
                    );

                    j_builder.create<mlir::scf::YieldOp>(j_loc);
                }
            );
            i_builder.create<mlir::scf::YieldOp>(i_loc);
        }
    );

    // 清理临时操作数
    if (!isStoredInSymbolTable(lhs)) {
        builder->create<mlir::memref::DeallocOp>(builder->getUnknownLoc(), lhs);
    }
    if (!isStoredInSymbolTable(rhs)) {
        builder->create<mlir::memref::DeallocOp>(builder->getUnknownLoc(), rhs);
    }

    return result;
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
                
            case ExprAST::Binary: {
                auto binary = static_cast<const BinaryExprAST*>(node);
                if (binary->getOp() == "-") {
                    auto lhs = generateMLIRForNode(binary->getLHS());
                    auto rhs = generateMLIRForNode(binary->getRHS());
                    if (!lhs || !rhs) return nullptr;
                    return generateTimeDiffMLIR(lhs, rhs);
                }
                return generateMLIRForBinary(binary);
            }

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
            case ExprAST::TimeCall: {
                auto timeCall = static_cast<const TimeCallExprAST*>(node);
                if (timeCall->getFuncName() == "now") {
                    return generateTimeNowMLIR(timeCall);
                }
                std::cerr << "Unknown time function: " << timeCall->getFuncName() << "\n";
                return nullptr;
            }
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
        
        // Declare runtime functions
        declareRuntimeFunctions();
        
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
        std::cerr << "Created main function with entry block\n";

        // Process AST nodes
        for (const auto& node : ast) {
            std::cerr << "\nProcessing node kind: " << node->getKind() << "\n";
            if (!generateMLIRForNode(node.get())) {
                std::cerr << "Failed to generate MLIR for node\n";
                module.dump();
                return nullptr;
            }
        }
        
        // Add return statement to main function
        auto zero = builder->create<mlir::arith::ConstantIntOp>(
            builder->getUnknownLoc(),
            0,
            builder->getI32Type()
        );
        
        // 创建 ValueRange 来包装返回值
        mlir::SmallVector<mlir::Value, 1> returnValues;
        returnValues.push_back(zero);
        
        // 使用 ValueRange 创建 ReturnOp
        builder->create<mlir::func::ReturnOp>(
            builder->getUnknownLoc(),
            returnValues
        );

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
    
    // Get dimensions from AST
    auto* rowsNum = static_cast<const NumberExprAST*>(expr->getRows());
    auto* colsNum = static_cast<const NumberExprAST*>(expr->getCols());
    
    int64_t rows = static_cast<int64_t>(rowsNum->getValue());
    int64_t cols = static_cast<int64_t>(colsNum->getValue());
    
    std::cerr << "[DEBUG] Creating tensor of size " << rows << "x" << cols << "\n";
    
    if (!validateMatrixDimensions(rows, cols, "random tensor")) {
        return nullptr;
    }
    
    // Create tensor type and allocation
    auto resultType = mlir::MemRefType::get({rows, cols}, builder->getF64Type());
    auto alloc = builder->create<mlir::memref::AllocOp>(builder->getUnknownLoc(), resultType);
    
    // Generate loop bounds
    auto lb = createConstantIndex(0);
    auto ub_rows = createConstantIndex(rows);
    auto ub_cols = createConstantIndex(cols);
    auto step = createConstantIndex(1);
    
    // Fill matrix with random values
    std::cerr << "[DEBUG] Generating random values\n";
    builder->create<mlir::scf::ForOp>(
        builder->getUnknownLoc(), lb, ub_rows, step,
        mlir::ValueRange{},
        [&](mlir::OpBuilder& i_builder, mlir::Location i_loc, mlir::Value i, mlir::ValueRange) {
            i_builder.create<mlir::scf::ForOp>(
                i_loc, lb, ub_cols, step,
                mlir::ValueRange{},
                [&](mlir::OpBuilder& j_builder, mlir::Location j_loc, mlir::Value j, mlir::ValueRange) {
                    auto random_val = j_builder.create<mlir::func::CallOp>(
                        j_loc,
                        "generate_random",
                        mlir::TypeRange{j_builder.getF64Type()},
                        mlir::ValueRange{}
                    );
                    
                    j_builder.create<mlir::memref::StoreOp>(
                        j_loc,
                        random_val.getResult(0),
                        alloc,
                        mlir::ValueRange{i, j}
                    );
                    
                    j_builder.create<mlir::scf::YieldOp>(j_loc);
                }
            );
            i_builder.create<mlir::scf::YieldOp>(i_loc);
        }
    );

    std::cerr << "[DEBUG] Tensor generation complete\n";
    return alloc;
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
    
    auto loc = builder->getUnknownLoc();
    
    // Handle different types
    if (auto numberExpr = llvm::dyn_cast<mlir::arith::ConstantFloatOp>(
            value.getDefiningOp())) {
        // For constant float values (like benchmark markers)
        builder->create<mlir::func::CallOp>(
            loc,
            "printFloat",
            mlir::TypeRange{},
            mlir::ValueRange{value}
        );
    } else if (auto intConst = llvm::dyn_cast<mlir::arith::ConstantIntOp>(
            value.getDefiningOp())) {
        // For constant integer values
        auto floatValue = builder->create<mlir::arith::SIToFPOp>(
            loc,
            builder->getF64Type(),
            value
        );
        
        builder->create<mlir::func::CallOp>(
            loc,
            "printFloat",
            mlir::TypeRange{},
            mlir::ValueRange{floatValue}
        );
    } else if (auto memrefTy = mlir::dyn_cast<mlir::MemRefType>(value.getType())) {
        // For matrices/tensors, print all values
        auto shape = memrefTy.getShape();
        int64_t rows = shape[0];
        int64_t cols = shape.size() > 1 ? shape[1] : 1;
        
        // Create nested loops to iterate through the matrix
        auto lb = builder->create<mlir::arith::ConstantIndexOp>(loc, 0);
        auto ub_rows = builder->create<mlir::arith::ConstantIndexOp>(loc, rows);
        auto ub_cols = builder->create<mlir::arith::ConstantIndexOp>(loc, cols);
        auto step = builder->create<mlir::arith::ConstantIndexOp>(loc, 1);
        
        // Outer loop for rows
        builder->create<mlir::scf::ForOp>(
            loc, lb, ub_rows, step,
            mlir::ValueRange{},
            [&](mlir::OpBuilder& nested, mlir::Location loc, mlir::Value i, mlir::ValueRange args) {
                // Inner loop for columns
                nested.create<mlir::scf::ForOp>(
                    loc, lb, ub_cols, step,
                    mlir::ValueRange{},
                    [&](mlir::OpBuilder& inner, mlir::Location loc, mlir::Value j, mlir::ValueRange inner_args) {
                        // Load value at current position
                        auto element = inner.create<mlir::memref::LoadOp>(
                            loc,
                            value,
                            mlir::ValueRange{i, j}
                        );
                        
                        // Print the element
                        inner.create<mlir::func::CallOp>(
                            loc,
                            "printFloat",
                            mlir::TypeRange{},
                            mlir::ValueRange{element}
                        );
                        
                        inner.create<mlir::scf::YieldOp>(loc);
                    }
                );
                nested.create<mlir::scf::YieldOp>(loc);
            }
        );
        
    } else if (auto floatTy = mlir::dyn_cast<mlir::FloatType>(value.getType())) {
        // For other float values
        builder->create<mlir::func::CallOp>(
            loc,
            "printFloat", 
            mlir::TypeRange{},
            mlir::ValueRange{value}
        );
    } else {
        std::cerr << "Unsupported type for print\n";
        return nullptr;
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

mlir::Value MLIRGen::generateTimeNowMLIR(const TimeCallExprAST* expr) {
    auto loc = builder->getUnknownLoc();
    
    // Declare system_time_msec function if not already declared
    if (!module.lookupSymbol("system_time_msec")) {
        auto funcType = mlir::FunctionType::get(
            context.get(),
            {},  // no input parameters
            {builder->getF64Type()}  // returns double
        );
        
        auto timeFunc = mlir::func::FuncOp::create(
            loc,
            "system_time_msec",
            funcType
        );
        
        timeFunc.setPrivate();
        module.push_back(timeFunc);
    }
    
    // Call time_msec function to get timestamp
    auto result = builder->create<mlir::func::CallOp>(
        loc,
        "system_time_msec",
        mlir::TypeRange{builder->getF64Type()},
        mlir::ValueRange{}
    );
    
    return result.getResult(0);
}

mlir::Value MLIRGen::generateTimeDiffMLIR(mlir::Value lhs, mlir::Value rhs) {
    auto loc = builder->getUnknownLoc();
    return builder->create<mlir::arith::SubFOp>(loc, lhs, rhs);
}

} // namespace matrix
