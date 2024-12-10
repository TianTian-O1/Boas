// MLIRGenNodes.cpp - AST节点处理实现
#include "mlirops/MLIRGen.h"

namespace matrix {

mlir::Value MLIRGen::generateMLIRForFunction(const FunctionAST* func) {
    if (!func) {
        std::cerr << "Error: Null function AST\n";
        return nullptr;
    }

    std::cerr << "[DEBUG] Generating MLIR for function: " << func->getName() << "\n";
    
    auto loc = builder->getUnknownLoc();

    // Create function type with appropriate return type and parameters
    std::vector<mlir::Type> paramTypes;
    
    // 获取函数参数类型
    for (const auto& arg : func->getArgs()) {
        paramTypes.push_back(builder->getIndexType());
    }
    
    // Create function type with parameters and return type
    auto returnMemRefType = mlir::MemRefType::get(
        {mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic}, 
        builder->getF64Type()
    );
    
    auto funcType = mlir::FunctionType::get(
        builder->getContext(), 
        paramTypes, 
        {returnMemRefType}
    );

    // Create or get existing function
    mlir::func::FuncOp funcOp;
    if (auto existingFunc = module.lookupSymbol<mlir::func::FuncOp>(func->getName())) {
        funcOp = existingFunc;
    } else {
        funcOp = mlir::func::FuncOp::create(loc, func->getName(), funcType);
        module.push_back(funcOp);
    }

    // Save current insertion point
    auto savedInsertionPoint = builder->saveInsertionPoint();

    // Create entry block with parameters
    auto* entryBlock = funcOp.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);

    // 保存当前作用域
    auto oldSymbolTable = symbolTable;
    
    // Add parameters to symbol table
    for (size_t i = 0; i < func->getArgs().size(); ++i) {
        symbolTable[func->getArgs()[i]] = entryBlock->getArgument(i);
    }

    // Process function body
    mlir::Value lastValue;
    bool hasExplicitReturn = false;
    
    for (const auto& stmt : func->getBody()) {
        auto result = generateMLIRForNode(stmt.get());
        
        if (result) {
            lastValue = result;
            
            // Handle assignments
            if (auto assign = llvm::dyn_cast<AssignmentExprAST>(stmt.get())) {
                symbolTable[assign->getName()] = result;
            }
            
            // Handle return statements
            if (llvm::isa<ReturnExprAST>(stmt.get())) {
                std::cerr << "[DEBUG] Generating return op for explicit return\n";
                if (!mlir::isa<mlir::MemRefType>(result.getType())) {
                    std::cerr << "Error: Function must return a matrix type\n";
                    return nullptr;
                }
                builder->create<mlir::func::ReturnOp>(loc, mlir::ValueRange{result});
                hasExplicitReturn = true;
                break;
            }
        }
    }

    // If no explicit return, return the last value or a default value
    if (!hasExplicitReturn) {
        std::cerr << "[DEBUG] Generating default return\n";
        mlir::Value returnValue;
        
        if (lastValue && mlir::isa<mlir::MemRefType>(lastValue.getType())) {
            returnValue = lastValue;
        } else {
            // Create default return value (1x1 zero matrix)
            auto one = createConstantIndex(1);
            auto empty = builder->create<mlir::memref::AllocOp>(
                loc,
                returnMemRefType,
                mlir::ValueRange{one, one}
            );
            
            auto zero = createConstantF64(0.0);
            auto idx = createConstantIndex(0);
            builder->create<mlir::memref::StoreOp>(
                loc,
                zero,
                empty,
                mlir::ValueRange{idx, idx}
            );
            
            returnValue = empty;
        }
        
        builder->create<mlir::func::ReturnOp>(loc, mlir::ValueRange{returnValue});
    }

    // 恢复作用域
    symbolTable = oldSymbolTable;

    // Restore insertion point
    builder->restoreInsertionPoint(savedInsertionPoint);

    // Return a dummy constant value since we've already added the function to the module
    return builder->create<mlir::arith::ConstantIntOp>(
        loc,
        0,
        32
    );
}

mlir::Value MLIRGen::generateNumberMLIR(const NumberExprAST* number) {
    return createConstantF64(number->getValue());
}

mlir::Value MLIRGen::generateMLIRForCall(const CallExprAST* expr) {
    auto loc = builder->getUnknownLoc();
    const std::string& callee = expr->getCallee();
    
    std::cerr << "[DEBUG] Generating call to function: " << callee << "\n";
    
    std::vector<mlir::Value> args;
    
    // Convert all arguments to the expected type (index)
    for (const auto& arg : expr->getArgs()) {
        auto argValue = generateMLIRForNode(arg.get());
        if (!argValue) {
            std::cerr << "Error: Failed to generate argument for function call\n";
            return nullptr;
        }
        
        std::cerr << "[DEBUG] Generated argument of type: ";
        argValue.getType().dump();
        std::cerr << "\n";
        
        // Convert argument to index type if needed
        if (!mlir::isa<mlir::IndexType>(argValue.getType())) {
            if (argValue.getType().isF64()) {
                // First convert float to integer
                auto intValue = builder->create<mlir::arith::FPToSIOp>(
                    loc,
                    builder->getI32Type(),
                    argValue
                );
                // Then convert integer to index
                argValue = builder->create<mlir::arith::IndexCastOp>(
                    loc,
                    builder->getIndexType(),
                    intValue
                );
                std::cerr << "[DEBUG] Converted f64 to index type\n";
            } else if (mlir::isa<mlir::IntegerType>(argValue.getType())) {
                argValue = builder->create<mlir::arith::IndexCastOp>(
                    loc,
                    builder->getIndexType(),
                    argValue
                );
                std::cerr << "[DEBUG] Converted integer to index type\n";
            }
        }
        args.push_back(argValue);
    }
    
    // Look up the function to determine its return type
    if (auto funcOp = module.lookupSymbol<mlir::func::FuncOp>(callee)) {
        std::cerr << "[DEBUG] Found function " << callee << " in module\n";
        auto returnTypes = funcOp.getFunctionType().getResults();
        auto callOp = builder->create<mlir::func::CallOp>(
            loc,
            callee,
            returnTypes,
            args
        );
        
        if (callOp.getNumResults() > 0) {
            auto result = callOp.getResult(0);
            std::cerr << "[DEBUG] Function call returned value of type: ";
            result.getType().dump();
            std::cerr << "\n";
            return result;
        }
        return nullptr;
    }
    
    std::cerr << "Error: Function " << callee << " not found in module\n";
    return nullptr;
}

mlir::Value MLIRGen::generateMLIRForArray(const ArrayExprAST* expr) {
    std::cerr << "[DEBUG] Generating Array operation\n";
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
    if (auto memrefTy = mlir::dyn_cast<mlir::MemRefType>(value.getType())) {
        // Convert matrix to unranked memref for printing
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
    } else if (value.getType().isF64() || mlir::isa<mlir::FloatType>(value.getType())) {
        builder->create<mlir::func::CallOp>(
            loc,
            "printFloat",
            mlir::TypeRange{},
            mlir::ValueRange{value}
        );
    } else if (mlir::isa<mlir::IntegerType>(value.getType()) || value.getType().isIndex()) {
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
    } else {
        std::cerr << "Unsupported type for print\n";
        return nullptr;
    }
    
    return value;
}

mlir::Value MLIRGen::generateMLIRForBinary(const BinaryExprAST* expr) {
    if (expr->getOp() == "matmul") {
        auto lhsValue = generateMLIRForNode(expr->getLHS());
        auto rhsValue = generateMLIRForNode(expr->getRHS());
        
        if (!lhsValue || !rhsValue) {
            std::cerr << "Failed to generate operands for matmul\n";
            return nullptr;
        }
        
        // Directly use the values for matrix multiplication
        auto loc = builder->getUnknownLoc();
        
        // Get dimensions of input matrices
        auto lhsType = mlir::dyn_cast<mlir::MemRefType>(lhsValue.getType());
        auto rhsType = mlir::dyn_cast<mlir::MemRefType>(rhsValue.getType());
        
        if (!lhsType || !rhsType) {
            std::cerr << "Matmul operands must be memref type\n";
            return nullptr;
        }
        
        // Get dimensions
        mlir::Value m = builder->create<mlir::memref::DimOp>(loc, lhsValue, 0);
        mlir::Value k1 = builder->create<mlir::memref::DimOp>(loc, lhsValue, 1);
        mlir::Value k2 = builder->create<mlir::memref::DimOp>(loc, rhsValue, 0);
        mlir::Value n = builder->create<mlir::memref::DimOp>(loc, rhsValue, 1);
        
        // Check dimension compatibility
        auto cmp = builder->create<mlir::arith::CmpIOp>(
            loc,
            mlir::arith::CmpIPredicate::ne,
            k1,
            k2
        );
        
        // Create result matrix
        auto resultType = mlir::MemRefType::get(
            {mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic},
            builder->getF64Type()
        );
        
        auto result = builder->create<mlir::memref::AllocOp>(
            loc,
            resultType,
            mlir::ValueRange{m, n}
        );
        
        // Initialize result to zero
        auto zero = createConstantF64(0.0);
        auto c0 = createConstantIndex(0);
        auto c1 = createConstantIndex(1);
        
        // Create nested loops for matrix multiplication
        auto i_loop = builder->create<mlir::scf::ForOp>(
            loc,
            c0,
            m,
            c1,
            mlir::ValueRange{},
            [&](mlir::OpBuilder& i_builder, mlir::Location i_loc, mlir::Value i, mlir::ValueRange) {
                i_builder.create<mlir::scf::ForOp>(
                    i_loc,
                    c0,
                    n,
                    c1,
                    mlir::ValueRange{},
                    [&](mlir::OpBuilder& j_builder, mlir::Location j_loc, mlir::Value j, mlir::ValueRange) {
                        // Initialize accumulator
                        auto init_sum = j_builder.create<mlir::arith::ConstantFloatOp>(
                            j_loc,
                            llvm::APFloat(0.0),
                            builder->getF64Type()
                        );
                        
                        // Inner product loop
                        auto sum = j_builder.create<mlir::scf::ForOp>(
                            j_loc,
                            c0,
                            k1,
                            c1,
                            mlir::ValueRange{init_sum},
                            [&](mlir::OpBuilder& k_builder, mlir::Location k_loc, mlir::Value k, mlir::ValueRange k_args) {
                                // Load elements
                                auto a_val = k_builder.create<mlir::memref::LoadOp>(
                                    k_loc,
                                    lhsValue,
                                    mlir::ValueRange{i, k}
                                );
                                auto b_val = k_builder.create<mlir::memref::LoadOp>(
                                    k_loc,
                                    rhsValue,
                                    mlir::ValueRange{k, j}
                                );
                                
                                // Multiply and add
                                auto mul = k_builder.create<mlir::arith::MulFOp>(k_loc, a_val, b_val);
                                auto new_sum = k_builder.create<mlir::arith::AddFOp>(k_loc, k_args[0], mul);
                                
                                k_builder.create<mlir::scf::YieldOp>(k_loc, mlir::ValueRange{new_sum});
                            }
                        );
                        
                        // Store result
                        j_builder.create<mlir::memref::StoreOp>(
                            j_loc,
                            sum.getResult(0),
                            result,
                            mlir::ValueRange{i, j}
                        );
                        
                        j_builder.create<mlir::scf::YieldOp>(j_loc);
                    }
                );
                i_builder.create<mlir::scf::YieldOp>(i_loc);
            }
        );
        
        return result;
    }
    
    return nullptr;
}

mlir::Value MLIRGen::generateMLIRForImport(const ImportAST* import) {
    if (!import) {
        std::cerr << "Error: Null import AST\n";
        return nullptr;
    }
    
    std::cerr << "[DEBUG] Processing import: " << import->getModuleName() << "\n";
    
    // For tensor module, we just need to ensure the tensor operations are available
    // No actual MLIR code needs to be generated for the import
    if (import->getModuleName() == "tensor") {
        std::cerr << "[DEBUG] Tensor module imported successfully\n";
        return builder->create<mlir::arith::ConstantIntOp>(
            builder->getUnknownLoc(),
            0,
            32
        );
    }
    
    std::cerr << "Error: Unknown module: " << import->getModuleName() << "\n";
    return nullptr;
}

mlir::Value MLIRGen::generateMLIRForVariable(const VariableExprAST* expr) {
    auto it = symbolTable.find(expr->getName());
    if (it != symbolTable.end()) {
        return it->second;
    }
    std::cerr << "Variable not found: " << expr->getName() << "\n";
    return nullptr;
}

mlir::Value MLIRGen::generateMLIRForAssignment(const AssignmentExprAST* expr) {
    auto value = generateMLIRForNode(expr->getValue());
    if (!value) return nullptr;
    
    symbolTable[expr->getName()] = value;
    return value;
}

} // namespace matrix