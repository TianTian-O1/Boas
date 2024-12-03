// MLIRGenMatrix.cpp - 矩阵运算相关实现
#include "mlirops/MLIRGen.h"

namespace matrix {


// MLIRGenMatrix.cpp 的实现
mlir::Value MLIRGen::generateMLIRForMatmul(const MatmulExprAST* expr) {
    std::cerr << "[DEBUG] Generating matrix multiplication\n";
    
    auto loc = builder->getUnknownLoc();
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
    
    // Get dimensions dynamically
    mlir::Value m = builder->create<mlir::memref::DimOp>(loc, lhs, 0);
    mlir::Value k1 = builder->create<mlir::memref::DimOp>(loc, lhs, 1);
    mlir::Value k2 = builder->create<mlir::memref::DimOp>(loc, rhs, 0);
    mlir::Value n = builder->create<mlir::memref::DimOp>(loc, rhs, 1);
    
    // Check dimension compatibility
    auto cmp = builder->create<mlir::arith::CmpIOp>(
        loc,
        mlir::arith::CmpIPredicate::ne,
        k1,
        k2
    );

    // Create constants for loops
    auto c0 = createConstantIndex(0);
    auto c1 = createConstantIndex(1);
    auto zero = createConstantF64(0.0);

    // Create result matrix with dynamic dimensions
    auto resultType = mlir::MemRefType::get(
        {mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic},
        builder->getF64Type()
    );
    
    auto result = builder->create<mlir::memref::AllocOp>(
        loc,
        resultType,
        mlir::ValueRange{m, n}
    );

    // Initialize result matrix to zero
    builder->create<mlir::scf::ForOp>(
        loc, c0, m, c1,
        mlir::ValueRange{},
        [&](mlir::OpBuilder& i_builder, mlir::Location i_loc, mlir::Value i, mlir::ValueRange) {
            i_builder.create<mlir::scf::ForOp>(
                i_loc, c0, n, c1,
                mlir::ValueRange{},
                [&](mlir::OpBuilder& j_builder, mlir::Location j_loc, mlir::Value j, mlir::ValueRange) {
                    j_builder.create<mlir::memref::StoreOp>(
                        j_loc,
                        zero,
                        result,
                        mlir::ValueRange{i, j}
                    );
                    j_builder.create<mlir::scf::YieldOp>(j_loc);
                }
            );
            i_builder.create<mlir::scf::YieldOp>(i_loc);
        }
    );

    // Choose optimization strategy
    mlir::Value optimizedResult;
    optimizedResult = createOptimizedMatmul(lhs, rhs, result, m, n, k1);

    // Clean up
    if (!isStoredInSymbolTable(lhs)) {
        builder->create<mlir::memref::DeallocOp>(loc, lhs);
    }
    if (!isStoredInSymbolTable(rhs)) {
        builder->create<mlir::memref::DeallocOp>(loc, rhs);
    }

    return optimizedResult ? optimizedResult : result;
}

mlir::Value MLIRGen::createOptimizedMatmul(mlir::Value lhs, mlir::Value rhs, mlir::Value result,
                                          mlir::Value M, mlir::Value N, mlir::Value K) {
    auto loc = builder->getUnknownLoc();
    auto c0 = createConstantIndex(0);
    auto c1 = createConstantIndex(1);
    auto zero = createConstantF64(0.0);

    // Main computation loops with dynamic bounds
    builder->create<mlir::scf::ForOp>(
        loc, c0, M, c1,
        mlir::ValueRange{},
        [&](mlir::OpBuilder& i_builder, mlir::Location i_loc, mlir::Value i, mlir::ValueRange) {
            i_builder.create<mlir::scf::ForOp>(
                i_loc, c0, N, c1,
                mlir::ValueRange{},
                [&](mlir::OpBuilder& j_builder, mlir::Location j_loc, mlir::Value j, mlir::ValueRange) {
                    auto inner_loop = j_builder.create<mlir::scf::ForOp>(
                        j_loc, c0, K, c1,
                        mlir::ValueRange{zero},
                        [&](mlir::OpBuilder& k_builder, mlir::Location k_loc, mlir::Value k, mlir::ValueRange k_args) {
                            auto a_val = k_builder.create<mlir::memref::LoadOp>(
                                k_loc,
                                lhs,
                                mlir::ValueRange{i, k}
                            );
                            
                            auto b_val = k_builder.create<mlir::memref::LoadOp>(
                                k_loc,
                                rhs,
                                mlir::ValueRange{k, j}
                            );

                            auto mul = k_builder.create<mlir::arith::MulFOp>(k_loc, a_val, b_val);
                            auto sum_iter = k_args[0];
                            auto new_sum = k_builder.create<mlir::arith::AddFOp>(k_loc, sum_iter, mul);

                            k_builder.create<mlir::scf::YieldOp>(k_loc, mlir::ValueRange{new_sum});
                        }
                    );

                    j_builder.create<mlir::memref::StoreOp>(
                        j_loc,
                        inner_loop.getResult(0),
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

mlir::Value MLIRGen::createBlockedMatmul(mlir::Value lhs, mlir::Value rhs, const MatmulExprAST* expr) {
    auto loc = builder->getUnknownLoc();
    
    // Get dynamic dimensions
    mlir::Value m = builder->create<mlir::memref::DimOp>(loc, lhs, 0);
    mlir::Value k = builder->create<mlir::memref::DimOp>(loc, lhs, 1);
    mlir::Value n = builder->create<mlir::memref::DimOp>(loc, rhs, 1);
    
    // Create result type with dynamic dimensions
    auto resultType = mlir::MemRefType::get(
        {mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic},
        builder->getF64Type()
    );
    
    auto result = builder->create<mlir::memref::AllocOp>(
        loc,
        resultType,
        mlir::ValueRange{m, n}
    );
    
    return createOptimizedMatmul(lhs, rhs, result, m, n, k);
}

mlir::Value MLIRGen::createVectorizedMatmul(mlir::Value lhs, mlir::Value rhs, const MatmulExprAST* expr) {
    auto loc = builder->getUnknownLoc();
    
    // Get dynamic dimensions
    mlir::Value m = builder->create<mlir::memref::DimOp>(loc, lhs, 0);
    mlir::Value k = builder->create<mlir::memref::DimOp>(loc, lhs, 1);
    mlir::Value n = builder->create<mlir::memref::DimOp>(loc, rhs, 1);
    
    // Create result type with dynamic dimensions
    auto resultType = mlir::MemRefType::get(
        {mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic},
        builder->getF64Type()
    );
    
    auto result = builder->create<mlir::memref::AllocOp>(
        loc,
        resultType,
        mlir::ValueRange{m, n}
    );
    
    return createOptimizedMatmul(lhs, rhs, result, m, n, k);
}


} // namespace matrix