// MLIRGenTensor.cpp - 张量操作相关实现
#include "mlirops/MLIRGen.h"
#include "mlirops/TensorMemoryPool.h"

namespace matrix {

mlir::Value MLIRGen::generateMLIRForTensor(const TensorExprAST* expr) {
    std::cerr << "[DEBUG] Generating Tensor operation\n";
    
    const auto& elements = expr->getElements();
    if (elements.empty()) {
        std::cerr << "Error: Empty tensor elements\n";
        return nullptr;
    }
    
    std::vector<mlir::Value> values;
    for (const auto& element : elements) {
        auto value = generateMLIRForNode(element.get());
        if (!value) {
            std::cerr << "Error: Failed to generate tensor element\n";
            return nullptr;
        }
        values.push_back(value);
    }
    
    auto elementType = builder->getF64Type();
    auto shape = mlir::MemRefType::get({static_cast<int64_t>(values.size())}, elementType);
    
    auto result = builder->create<mlir::memref::AllocOp>(
        builder->getUnknownLoc(),
        shape
    );
    
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
    
    // 获取数组值
    auto arrayExpr = static_cast<const ArrayExprAST*>(expr->getValues());
    const auto& values = arrayExpr->getElements();
    int idx = 0;
    
    for (int i = 0; i < rowsNum; i++) {
        for (int j = 0; j < colsNum; j++) {
            if (idx >= values.size()) {
                llvm::errs() << "Error: Not enough values for tensor dimensions\n";
                return nullptr;
            }
            
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

mlir::Value MLIRGen::generateMLIRForTensorRandom(const TensorRandomExprAST* expr) {
    std::cerr << "[DEBUG] Generating random tensor with dynamic dimensions\n";
    
    auto loc = builder->getUnknownLoc();
    
    // Get row and column values
    mlir::Value rowValue, colValue;
    
    // Handle row dimension
    if (auto* rowNum = llvm::dyn_cast<NumberExprAST>(expr->getRows())) {
        rowValue = createConstantIndex(static_cast<int64_t>(rowNum->getValue()));
        std::cerr << "[DEBUG] Row dimension from number: " << rowNum->getValue() << "\n";
    } else if (auto* rowVar = llvm::dyn_cast<VariableExprAST>(expr->getRows())) {
        auto it = symbolTable.find(rowVar->getName());
        if (it != symbolTable.end()) {
            rowValue = it->second;
            std::cerr << "[DEBUG] Row dimension from variable: " << rowVar->getName() << "\n";
        } else {
            rowValue = createConstantIndex(4); // Default value
            symbolTable[rowVar->getName()] = rowValue;
            std::cerr << "[DEBUG] Created default row dimension: 4\n";
        }
    }
    
    // Handle column dimension
    if (auto* colNum = llvm::dyn_cast<NumberExprAST>(expr->getCols())) {
        colValue = createConstantIndex(static_cast<int64_t>(colNum->getValue()));
        std::cerr << "[DEBUG] Col dimension from number: " << colNum->getValue() << "\n";
    } else if (auto* colVar = llvm::dyn_cast<VariableExprAST>(expr->getCols())) {
        auto it = symbolTable.find(colVar->getName());
        if (it != symbolTable.end()) {
            colValue = it->second;
            std::cerr << "[DEBUG] Col dimension from variable: " << colVar->getName() << "\n";
        } else {
            colValue = createConstantIndex(4); // Default value
            symbolTable[colVar->getName()] = colValue;
            std::cerr << "[DEBUG] Created default col dimension: 4\n";
        }
    }

    // Create memref type with dynamic dimensions
    auto memrefType = mlir::MemRefType::get(
        {mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic},
        builder->getF64Type()
    );

    // Allocate matrix with dynamic dimensions
    auto result = builder->create<mlir::memref::AllocOp>(
        loc,
        memrefType,
        mlir::ValueRange{rowValue, colValue}
    );

    // Calculate total size and batch size
    auto totalSize = builder->create<mlir::arith::MulIOp>(loc, rowValue, colValue);
    auto batchSize = createConstantIndex(32);
    auto actualBatchSize = builder->create<mlir::arith::MinSIOp>(loc, batchSize, totalSize);

    // Create random buffer with dynamic size
    auto bufferType = mlir::MemRefType::get({mlir::ShapedType::kDynamic}, builder->getF64Type());
    auto randomBuffer = builder->create<mlir::memref::AllocOp>(
        loc,
        bufferType,
        mlir::ValueRange{actualBatchSize}
    );

    // Declare random generation function if not exists
    mlir::func::FuncOp genRandomFunc;
    if (!(genRandomFunc = module.lookupSymbol<mlir::func::FuncOp>("generate_random"))) {
        auto funcType = mlir::FunctionType::get(
            builder->getContext(),
            {},
            {builder->getF64Type()}
        );
        genRandomFunc = mlir::func::FuncOp::create(loc, "generate_random", funcType);
        genRandomFunc.setPrivate();
        mlir::SymbolTable::setSymbolVisibility(genRandomFunc, mlir::SymbolTable::Visibility::Private);
        module.push_back(genRandomFunc);
    }

    // Fill random buffer with batched random values
    builder->create<mlir::scf::ForOp>(
        loc,
        createConstantIndex(0),
        actualBatchSize,
        createConstantIndex(1),
        mlir::ValueRange{},
        [&](mlir::OpBuilder& builder, mlir::Location loc, mlir::Value idx, mlir::ValueRange args) {
            // Generate random value
            auto randVal = builder.create<mlir::func::CallOp>(
                loc,
                genRandomFunc,
                mlir::ValueRange{}
            ).getResult(0);
            
            // Store in buffer
            builder.create<mlir::memref::StoreOp>(
                loc,
                randVal,
                randomBuffer,
                mlir::ValueRange{idx}
            );
            
            builder.create<mlir::scf::YieldOp>(loc);
        }
    );

    // Fill result matrix using random buffer values
    builder->create<mlir::scf::ForOp>(
        loc,
        createConstantIndex(0),
        totalSize,
        createConstantIndex(1),
        mlir::ValueRange{},
        [&](mlir::OpBuilder& builder, mlir::Location loc, mlir::Value idx, mlir::ValueRange args) {
            // Calculate row and column indices
            auto i = builder.create<mlir::arith::DivUIOp>(loc, idx, colValue);
            auto j = builder.create<mlir::arith::RemUIOp>(loc, idx, colValue);
            
            // Calculate buffer index (cycle through buffer)
            auto bufferIdx = builder.create<mlir::arith::RemUIOp>(
                loc,
                idx,
                actualBatchSize
            );
            
            // Load random value from buffer
            auto randVal = builder.create<mlir::memref::LoadOp>(
                loc,
                randomBuffer,
                mlir::ValueRange{bufferIdx}
            );
            
            // Store value in result matrix
            builder.create<mlir::memref::StoreOp>(
                loc,
                randVal,
                result,
                mlir::ValueRange{i, j}
            );
            
            builder.create<mlir::scf::YieldOp>(loc);
        }
    );

    // Clean up: deallocate random buffer
    builder->create<mlir::memref::DeallocOp>(loc, randomBuffer);
    
    std::cerr << "[DEBUG] Dynamic tensor generation complete\n";
    return result;
}


} // namespace matrix