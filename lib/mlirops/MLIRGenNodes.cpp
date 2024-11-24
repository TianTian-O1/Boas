// MLIRGenNodes.cpp - AST节点处理实现
#include "mlirops/MLIRGen.h"

namespace matrix {


// MLIRGenNodes.cpp

mlir::Value MLIRGen::generateMLIRForFunction(const FunctionAST* func) {
    if (!func) {
        std::cerr << "Error: Null function AST\n";
        return nullptr;
    }

    std::cerr << "[DEBUG] Generating MLIR for function: " << func->getName() << "\n";
    
    auto loc = builder->getUnknownLoc();

    // Create function type with appropriate return type and parameters
    mlir::Type returnType;
    std::vector<mlir::Type> paramTypes;
    
    if (func->getName() == "benchmark") {
        // benchmark function returns dynamic matrix and takes size parameter
        returnType = mlir::MemRefType::get(
            {mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic},
            builder->getF64Type()
        );
        paramTypes.push_back(builder->getIndexType()); // Add size parameter
    } else {
        // other functions return f64
        returnType = builder->getF64Type();
    }

    // Create function type with parameters
    auto funcType = mlir::FunctionType::get(builder->getContext(), paramTypes, {returnType});
    mlir::func::FuncOp funcOp;
    
    if (auto existingFunc = module.lookupSymbol<mlir::func::FuncOp>(func->getName())) {
        funcOp = existingFunc;
    } else {
        funcOp = mlir::func::FuncOp::create(loc, func->getName(), funcType);
        module.push_back(funcOp);
    }

    // Create entry block with parameters
    auto* entryBlock = funcOp.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);

    // Save symbol table state
    std::map<std::string, mlir::Value> oldSymbols = symbolTable;

    // Add parameters to symbol table if any
    if (func->getName() == "benchmark" && entryBlock->getNumArguments() > 0) {
        symbolTable["size"] = entryBlock->getArgument(0);
    }

    // Process function body
    mlir::Value result;
    for (const auto& stmt : func->getBody()) {
        result = generateMLIRForNode(stmt.get());
        if (!result) {
            std::cerr << "[DEBUG] Statement generated null value\n";
            continue;
        }
    }

    // Handle return value
    if (result) {
        if (func->getName() == "benchmark") {
            // For benchmark function
            if (!mlir::isa<mlir::MemRefType>(result.getType())) {
                std::cerr << "Error: Benchmark function must return memref type\n";
                return nullptr;
            }

            // Create a copy of the result to ensure proper memory management
            auto memrefType = mlir::cast<mlir::MemRefType>(result.getType());
            auto m = builder->create<mlir::memref::DimOp>(loc, result, 0);
            auto n = builder->create<mlir::memref::DimOp>(loc, result, 1);
            
            // Allocate new memory for the return value
            auto returnMemref = builder->create<mlir::memref::AllocOp>(
                loc,
                memrefType,
                mlir::ValueRange{m, n}
            );
            
            // Copy data to the new memory
            auto c0 = createConstantIndex(0);
            auto c1 = createConstantIndex(1);
            
            builder->create<mlir::scf::ForOp>(
                loc, c0, m, c1,
                mlir::ValueRange{},
                [&](mlir::OpBuilder& i_builder, mlir::Location i_loc, mlir::Value i, mlir::ValueRange) {
                    i_builder.create<mlir::scf::ForOp>(
                        i_loc, c0, n, c1,
                        mlir::ValueRange{},
                        [&](mlir::OpBuilder& j_builder, mlir::Location j_loc, mlir::Value j, mlir::ValueRange) {
                            // Load value from source
                            auto val = j_builder.create<mlir::memref::LoadOp>(
                                j_loc,
                                result,
                                mlir::ValueRange{i, j}
                            );
                            
                            // Store value to destination
                            j_builder.create<mlir::memref::StoreOp>(
                                j_loc,
                                val,
                                returnMemref,
                                mlir::ValueRange{i, j}
                            );
                            
                            j_builder.create<mlir::scf::YieldOp>(j_loc);
                        }
                    );
                    i_builder.create<mlir::scf::YieldOp>(i_loc);
                }
            );
            
            // Deallocate the original result if it's a temporary
            if (!isStoredInSymbolTable(result)) {
                builder->create<mlir::memref::DeallocOp>(loc, result);
            }
            
            // Return the new memory
            builder->create<mlir::func::ReturnOp>(loc, mlir::ValueRange{returnMemref});
        } else {
            // For other functions
            mlir::Value returnValue;
            if (mlir::isa<mlir::MemRefType>(result.getType())) {
                // Get first element of matrix
                auto zero = createConstantIndex(0);
                returnValue = builder->create<mlir::memref::LoadOp>(
                    loc,
                    result,
                    mlir::ValueRange{zero, zero}
                );
                
                // Clean up temporary matrix
                if (!isStoredInSymbolTable(result)) {
                    builder->create<mlir::memref::DeallocOp>(loc, result);
                }
            } else if (result.getType().isF64()) {
                returnValue = result;
            } else if (mlir::isa<mlir::IndexType>(result.getType())) {
                returnValue = builder->create<mlir::arith::SIToFPOp>(
                    loc,
                    builder->getF64Type(),
                    result
                );
            } else {
                returnValue = builder->create<mlir::arith::ConstantFloatOp>(
                    loc,
                    llvm::APFloat(0.0),
                    builder->getF64Type()
                );
            }
            builder->create<mlir::func::ReturnOp>(loc, mlir::ValueRange{returnValue});
        }
    } else {
        // Handle default return values
        if (func->getName() == "benchmark") {
            auto size = entryBlock->getArgument(0);
            auto memrefType = mlir::MemRefType::get(
                {mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic},
                builder->getF64Type()
            );
            auto empty = builder->create<mlir::memref::AllocOp>(
                loc,
                memrefType,
                mlir::ValueRange{size, size}
            );
            
            // Initialize to zeros
            initializeMemRefToZero(empty, size, size);
            
            builder->create<mlir::func::ReturnOp>(loc, mlir::ValueRange{empty});
        } else {
            auto zero = builder->create<mlir::arith::ConstantFloatOp>(
                loc,
                llvm::APFloat(0.0),
                builder->getF64Type()
            );
            builder->create<mlir::func::ReturnOp>(loc, mlir::ValueRange{zero});
        }
    }

    // Restore symbol table
    symbolTable = oldSymbols;

    return result;
}

// Helper function to initialize memref to zero
void MLIRGen::initializeMemRefToZero(mlir::Value memref, mlir::Value rows, mlir::Value cols) {
    auto loc = builder->getUnknownLoc();
    auto c0 = createConstantIndex(0);
    auto c1 = createConstantIndex(1);
    auto zero = createConstantF64(0.0);
    
    builder->create<mlir::scf::ForOp>(
        loc,
        c0,
        rows,
        c1,
        mlir::ValueRange{},
        [&](mlir::OpBuilder& i_builder, mlir::Location i_loc, mlir::Value i, mlir::ValueRange) {
            i_builder.create<mlir::scf::ForOp>(
                i_loc,
                c0,
                cols,
                c1,
                mlir::ValueRange{},
                [&](mlir::OpBuilder& j_builder, mlir::Location j_loc, mlir::Value j, mlir::ValueRange) {
                    j_builder.create<mlir::memref::StoreOp>(
                        j_loc,
                        zero,
                        memref,
                        mlir::ValueRange{i, j}
                    );
                    j_builder.create<mlir::scf::YieldOp>(j_loc);
                }
            );
            i_builder.create<mlir::scf::YieldOp>(i_loc);
        }
    );
}

void MLIRGen::handleFunctionReturn(const FunctionAST* func, mlir::Value result, mlir::Block* entryBlock) {
    auto loc = builder->getUnknownLoc();
    
    if (result) {
        if (func->getName() == "benchmark") {
            // For benchmark function, return memref directly
            if (!mlir::isa<mlir::MemRefType>(result.getType())) {
                std::cerr << "Error: Benchmark function must return memref type\n";
                return;
            }
            builder->create<mlir::func::ReturnOp>(loc, mlir::ValueRange{result});
        } else {
            // For other functions, ensure f64 return type
            mlir::Value returnValue;
            if (mlir::isa<mlir::MemRefType>(result.getType())) {
                // If result is a matrix, return the first element or 0.0
                auto zero = createConstantIndex(0);
                auto value = builder->create<mlir::memref::LoadOp>(
                    loc,
                    result,
                    mlir::ValueRange{zero, zero}
                );
                returnValue = value;
            } else if (result.getType().isF64()) {
                returnValue = result;
            } else if (mlir::isa<mlir::IndexType>(result.getType())) {
                // Convert index to f64
                returnValue = builder->create<mlir::arith::SIToFPOp>(
                    loc,
                    builder->getF64Type(),
                    result
                );
            } else {
                // Default to 0.0 for unsupported types
                returnValue = builder->create<mlir::arith::ConstantFloatOp>(
                    loc,
                    llvm::APFloat(0.0),
                    builder->getF64Type()
                );
            }
            builder->create<mlir::func::ReturnOp>(loc, mlir::ValueRange{returnValue});
        }
    } else {
        // Handle default return values
        handleDefaultReturn(func, entryBlock);
    }
}

void MLIRGen::handleDefaultReturn(const FunctionAST* func, mlir::Block* entryBlock) {
    auto loc = builder->getUnknownLoc();
    
    if (func->getName() == "benchmark") {
        // Get size from parameter
        auto size = entryBlock->getArgument(0);
        
        // Create default square matrix of the given size
        auto memrefType = mlir::MemRefType::get(
            {mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic},
            builder->getF64Type()
        );
        auto empty = builder->create<mlir::memref::AllocOp>(
            loc,
            memrefType,
            mlir::ValueRange{size, size}
        );
        
        // Initialize to zeros
        auto c0 = createConstantIndex(0);
        auto c1 = createConstantIndex(1);
        auto zero = createConstantF64(0.0);
        
        builder->create<mlir::scf::ForOp>(
            loc,
            c0,
            size,
            c1,
            mlir::ValueRange{},
            [&](mlir::OpBuilder& i_builder, mlir::Location i_loc, mlir::Value i, mlir::ValueRange) {
                i_builder.create<mlir::scf::ForOp>(
                    i_loc,
                    c0,
                    size,
                    c1,
                    mlir::ValueRange{},
                    [&](mlir::OpBuilder& j_builder, mlir::Location j_loc, mlir::Value j, mlir::ValueRange) {
                        j_builder.create<mlir::memref::StoreOp>(
                            j_loc,
                            zero,
                            empty,
                            mlir::ValueRange{i, j}
                        );
                        j_builder.create<mlir::scf::YieldOp>(j_loc);
                    }
                );
                i_builder.create<mlir::scf::YieldOp>(i_loc);
            }
        );
        
        builder->create<mlir::func::ReturnOp>(loc, mlir::ValueRange{empty});
    } else {
        // Return 0.0 for regular functions
        auto zero = builder->create<mlir::arith::ConstantFloatOp>(
            loc,
            llvm::APFloat(0.0),
            builder->getF64Type()
        );
        builder->create<mlir::func::ReturnOp>(loc, mlir::ValueRange{zero});
    }
}

mlir::Value MLIRGen::generateMLIRForCall(const CallExprAST* expr) {
    auto loc = builder->getUnknownLoc();
    const std::string& callee = expr->getCallee();
    
    // Special handling for benchmark function
    if (callee == "benchmark") {
        std::vector<mlir::Value> args;
        
        // Convert all arguments to the expected type (index)
        for (const auto& arg : expr->getArgs()) {
            auto argValue = generateMLIRForNode(arg.get());
            if (!argValue) {
                std::cerr << "Error: Failed to generate argument for benchmark\n";
                return nullptr;
            }
            
            // Convert argument to index type if needed
            if (!mlir::isa<mlir::IndexType>(argValue.getType())) {
                if (mlir::isa<mlir::IntegerType>(argValue.getType()) || 
                    argValue.getType().isIndex()) {
                    argValue = builder->create<mlir::arith::IndexCastOp>(
                        loc,
                        builder->getIndexType(),
                        argValue
                    );
                } else {
                    std::cerr << "Error: Benchmark argument must be convertible to index\n";
                    return nullptr;
                }
            }
            args.push_back(argValue);
        }
        
        // Create memref type for return value
        auto returnType = mlir::MemRefType::get(
            {mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic},
            builder->getF64Type()
        );
        
        // Create function call
        auto funcType = mlir::FunctionType::get(
            builder->getContext(),
            {builder->getIndexType()},
            {returnType}
        );
        
        auto callOp = builder->create<mlir::func::CallOp>(
            loc,
            callee,
            funcType.getResults(),
            args
        );
        
        return callOp.getResult(0);
    } 
    
    // Handle other function calls normally
    std::vector<mlir::Value> args;
    for (const auto& arg : expr->getArgs()) {
        auto argValue = generateMLIRForNode(arg.get());
        if (!argValue) {
            std::cerr << "Error: Failed to generate argument\n";
            return nullptr;
        }
        args.push_back(argValue);
    }
    
    auto funcType = mlir::FunctionType::get(
        builder->getContext(),
        {},
        {builder->getF64Type()}
    );
    
    return builder->create<mlir::func::CallOp>(
        loc,
        callee,
        funcType.getResults(),
        args
    ).getResult(0);
}

mlir::Value MLIRGen::generateMLIRForImport(const ImportAST* import) {
    if (!import) {
        std::cerr << "Error: Null import AST\n";
        return nullptr;
    }

    std::cerr << "[DEBUG] Processing import: " << import->getModuleName() << "\n";
    
    return builder->create<mlir::arith::ConstantIntOp>(
        builder->getUnknownLoc(),
        0,
        32
    );
}

mlir::Value MLIRGen::generateMLIRForVariable(const VariableExprAST* expr) {
    auto it = symbolTable.find(expr->getName());
    if (it != symbolTable.end()) {
        return it->second;
    }
    
    llvm::errs() << "[DEBUG] Variable " << expr->getName() << " not found in symbol table\n";
    return nullptr;
}

mlir::Value MLIRGen::generateMLIRForAssignment(const AssignmentExprAST* expr) {
    auto value = generateMLIRForNode(expr->getValue());
    if (!value) return nullptr;
    
    // 如果是数字类型，确保转换为正确的 MLIR 类型
    if (auto* numExpr = llvm::dyn_cast<NumberExprAST>(expr->getValue())) {
        if (value.getType().isF64()) {
            // 对于矩阵维度，我们需要将浮点数转换为整数
            auto intValue = builder->create<mlir::arith::FPToSIOp>(
                builder->getUnknownLoc(),
                builder->getI32Type(),
                value
            );
            symbolTable[expr->getName()] = intValue;
            return intValue;
        }
    }
    
    symbolTable[expr->getName()] = value;
    return value;
}

mlir::Value MLIRGen::generateNumberMLIR(const NumberExprAST* number) {
    return createConstantF64(number->getValue());
}

mlir::Value MLIRGen::generateMLIRForBinary(const BinaryExprAST* expr) {
    llvm::errs() << "[DEBUG] Processing Binary node\n";
    llvm::errs() << "[DEBUG] Binary operator: '" << expr->getOp() << "'\n";
    
    if (expr->getOp() == "matmul") {
        llvm::errs() << "[DEBUG] Found matmul operation\n";
        
        auto* lhs = static_cast<const VariableExprAST*>(expr->getLHS());
        auto* rhs = static_cast<const VariableExprAST*>(expr->getRHS());
        
        llvm::errs() << "[DEBUG] Matmul operands: " << lhs->getName() << " * " << rhs->getName() << "\n";
        
        auto lhsValue = symbolTable[lhs->getName()];
        auto rhsValue = symbolTable[rhs->getName()];
        
        if (!lhsValue || !rhsValue) {
            llvm::errs() << "[DEBUG] Failed to find operands in symbol table\n";
            return nullptr;
        }
        
        auto matmulExpr = std::make_unique<MatmulExprAST>(
            std::make_unique<VariableExprAST>(lhs->getName()),
            std::make_unique<VariableExprAST>(rhs->getName())
        );
        
        auto result = generateMLIRForMatmul(matmulExpr.get());
        
        if (!result || !mlir::isa<mlir::MemRefType>(result.getType())) {
            llvm::errs() << "[DEBUG] Matmul result must be memref type\n";
            return nullptr;
        }
        
        return result;
    }
    
    return nullptr;
}

mlir::Value MLIRGen::generateMLIRForArray(const ArrayExprAST* expr) {
    std::cerr << "[DEBUG] Generating Array operation\n";
    // Implementation for array operations - to be implemented
    return nullptr;
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
        // 将结果矩阵转换为 unranked memref 并打印
        auto resultType = mlir::UnrankedMemRefType::get(
            builder->getF64Type(),
            0
        );
        auto cast = builder->create<mlir::memref::CastOp>(
            loc,
            resultType,
            value
        );
        
        builder->create<mlir::func::CallOp>(
            loc,
            "printMemrefF64",
            mlir::TypeRange{},
            mlir::ValueRange{cast}
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


} // namespace matrix