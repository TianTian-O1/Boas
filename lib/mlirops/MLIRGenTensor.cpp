// MLIRGenTensor.cpp - 张量操作相关实现
#include "mlirops/MLIRGen.h"

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

mlir::Value MLIRGen::generateMLIRForTensorRandom(const TensorRandomExprAST* expr) {
    std::cerr << "[DEBUG] Generating random tensor\n";
    
    auto* rowsNum = static_cast<const NumberExprAST*>(expr->getRows());
    auto* colsNum = static_cast<const NumberExprAST*>(expr->getCols());
    
    int64_t rows = static_cast<int64_t>(rowsNum->getValue());
    int64_t cols = static_cast<int64_t>(colsNum->getValue());
    
    std::cerr << "[DEBUG] Creating tensor of size " << rows << "x" << cols << "\n";
    
    if (rows <= 0 || cols <= 0 || rows > 2048 || cols > 2048) {
        std::cerr << "Invalid dimensions for random tensor: " << rows << "x" << cols << "\n";
        return nullptr;
    }
    
    auto resultType = mlir::MemRefType::get({rows, cols}, builder->getF64Type());
    auto alloc = builder->create<mlir::memref::AllocOp>(builder->getUnknownLoc(), resultType);
    
    auto lb = createConstantIndex(0);
    auto ub_rows = createConstantIndex(rows);
    auto ub_cols = createConstantIndex(cols);
    auto step = createConstantIndex(1);
    
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

} // namespace matrix
