// MLIRGenMatrix.cpp - 矩阵运算相关实现
#include "mlirops/MLIRGen.h"

namespace matrix {

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

    // 创建常量
    auto c0 = createConstantIndex(0);
    auto c1 = createConstantIndex(1);
    auto cM = createConstantIndex(M);
    auto cK = createConstantIndex(K);
    auto cN = createConstantIndex(N);
    auto zero = createConstantF64(0.0);

    // 创建结果矩阵
    auto resultType = mlir::MemRefType::get({M, N}, builder->getF64Type());
    auto result = builder->create<mlir::memref::AllocOp>(builder->getUnknownLoc(), resultType);

    // 初始化结果矩阵为0
    builder->create<mlir::scf::ForOp>(
        builder->getUnknownLoc(), c0, cM, c1,
        mlir::ValueRange{},
        [&](mlir::OpBuilder& i_builder, mlir::Location i_loc, mlir::Value i, mlir::ValueRange) {
            i_builder.create<mlir::scf::ForOp>(
                i_loc, c0, cN, c1,
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

    // 根据矩阵大小选择不同的优化策略
    mlir::Value optimizedResult;
    if (M >= TILE_SIZE && N >= TILE_SIZE && K >= TILE_SIZE) {
        // 对于较大的矩阵使用分块优化
        optimizedResult = createBlockedMatmul(lhs, rhs, expr);
    } else if (M >= VECTOR_SIZE && N >= VECTOR_SIZE && K >= VECTOR_SIZE) {
        // 对于中等大小的矩阵使用向量化
        optimizedResult = createVectorizedMatmul(lhs, rhs, expr);
    } else {
        // 对于小矩阵使用基本实现
        optimizedResult = createOptimizedMatmul(lhs, rhs, result, M, N, K);
    }

    // 清理原始操作数
    if (!isStoredInSymbolTable(lhs)) {
        builder->create<mlir::memref::DeallocOp>(builder->getUnknownLoc(), lhs);
    }
    if (!isStoredInSymbolTable(rhs)) {
        builder->create<mlir::memref::DeallocOp>(builder->getUnknownLoc(), rhs);
    }

    return optimizedResult ? optimizedResult : result;
}

mlir::Value MLIRGen::createOptimizedMatmul(mlir::Value lhs, mlir::Value rhs, mlir::Value result,
                                          int64_t M, int64_t N, int64_t K) {
    auto c0 = createConstantIndex(0);
    auto c1 = createConstantIndex(1);
    auto cM = createConstantIndex(M);
    auto cK = createConstantIndex(K);
    auto cN = createConstantIndex(N);
    auto zero = createConstantF64(0.0);

    // 主计算循环
    builder->create<mlir::scf::ForOp>(
        builder->getUnknownLoc(), c0, cM, c1,
        mlir::ValueRange{},
        [&](mlir::OpBuilder& i_builder, mlir::Location i_loc, mlir::Value i, mlir::ValueRange) {
            i_builder.create<mlir::scf::ForOp>(
                i_loc, c0, cN, c1,
                mlir::ValueRange{},
                [&](mlir::OpBuilder& j_builder, mlir::Location j_loc, mlir::Value j, mlir::ValueRange) {
                    auto inner_loop = j_builder.create<mlir::scf::ForOp>(
                        j_loc, c0, cK, c1,
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
    // 实现分块矩阵乘法
    // TODO: 使用 tile size 进行分块计算
    // 暂时返回基本实现
    auto lhsType = mlir::dyn_cast<mlir::MemRefType>(lhs.getType());
    auto rhsType = mlir::dyn_cast<mlir::MemRefType>(rhs.getType());
    
    int64_t M = lhsType.getShape()[0];
    int64_t K = lhsType.getShape()[1];
    int64_t N = rhsType.getShape()[1];
    
    auto resultType = mlir::MemRefType::get({M, N}, builder->getF64Type());
    auto result = builder->create<mlir::memref::AllocOp>(builder->getUnknownLoc(), resultType);
    
    return createOptimizedMatmul(lhs, rhs, result, M, N, K);
}

mlir::Value MLIRGen::createVectorizedMatmul(mlir::Value lhs, mlir::Value rhs, const MatmulExprAST* expr) {
    // 实现向量化矩阵乘法
    // TODO: 使用 vector size 进行向量化计算
    // 暂时返回基本实现
    auto lhsType = mlir::dyn_cast<mlir::MemRefType>(lhs.getType());
    auto rhsType = mlir::dyn_cast<mlir::MemRefType>(rhs.getType());
    
    int64_t M = lhsType.getShape()[0];
    int64_t K = lhsType.getShape()[1];
    int64_t N = rhsType.getShape()[1];
    
    auto resultType = mlir::MemRefType::get({M, N}, builder->getF64Type());
    auto result = builder->create<mlir::memref::AllocOp>(builder->getUnknownLoc(), resultType);
    
    return createOptimizedMatmul(lhs, rhs, result, M, N, K);
}

} // namespace matrix