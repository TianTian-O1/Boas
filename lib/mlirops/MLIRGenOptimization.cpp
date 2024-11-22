// MLIRGenOptimizations.cpp - 优化相关实现
#include "mlirops/MLIRGen.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

namespace matrix {

mlir::Value MLIRGen::createFullyUnrolledMatmul(mlir::Value lhs, mlir::Value rhs,
                                              mlir::Value result,
                                              int64_t M, int64_t N, int64_t K) {
    auto loc = builder->getUnknownLoc();
    
    // 完全展开的矩阵乘法实现
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            // 初始化累加器
            auto acc = createConstantF64(0.0);
            
            for (int k = 0; k < K; k++) {
                auto a = builder->create<mlir::memref::LoadOp>(
                    loc, lhs, 
                    mlir::ValueRange{createConstantIndex(i), createConstantIndex(k)});
                    
                auto b = builder->create<mlir::memref::LoadOp>(
                    loc, rhs,
                    mlir::ValueRange{createConstantIndex(k), createConstantIndex(j)});
                    
                auto mul = builder->create<mlir::arith::MulFOp>(loc, a, b);
                acc = builder->create<mlir::arith::AddFOp>(loc, acc, mul);
            }
            
            builder->create<mlir::memref::StoreOp>(
                loc, acc, result,
                mlir::ValueRange{createConstantIndex(i), createConstantIndex(j)});
        }
    }
    
    return result;
}

void MLIRGen::addMemoryAccessAttributes(mlir::Operation* op) {
    if (!op) return;
    
    // 添加内存访问相关的优化属性
    op->setAttr("memory_access.pattern", 
                builder->getStringAttr("contiguous"));
    op->setAttr("memory_access.alignment", 
                builder->getI64IntegerAttr(32));  // 假设 32 字节对齐
}

mlir::Value MLIRGen::optimizeSmallMatrixMultiply(mlir::Value lhs, mlir::Value rhs, mlir::Value result,
                                                int64_t M, int64_t N, int64_t K) {
    auto loc = builder->getUnknownLoc();
    
    // 对于非常小的矩阵（8x8及以下），完全展开
    if (M <= 8 && N <= 8 && K <= 8) {
        return createFullyUnrolledMatmul(lhs, rhs, result, M, N, K);
    }
    
    // 对于小矩阵（64x64及以下），使用寄存器分块+向量化
    if (M <= 64 && N <= 64 && K <= 64) {
        const int REGISTER_BLOCK_M = 4;
        const int REGISTER_BLOCK_N = 16;
        const int REGISTER_BLOCK_K = 4;
        
        // 创建向量类型
        auto vecType = mlir::VectorType::get({REGISTER_BLOCK_N}, builder->getF64Type());
        
        for (int i = 0; i < M; i += REGISTER_BLOCK_M) {
            for (int j = 0; j < N; j += REGISTER_BLOCK_N) {
                // 初始化累加器
                auto zero = createConstantF64(0.0);
                mlir::Value accum = builder->create<mlir::vector::BroadcastOp>(
                    loc, vecType, zero).getResult();
                
                for (int k = 0; k < K; k += REGISTER_BLOCK_K) {
                    // 加载数据
                    auto vecA = builder->create<mlir::vector::LoadOp>(
                        loc, vecType, lhs,
                        mlir::ValueRange{createConstantIndex(i), createConstantIndex(k)}).getResult();
                        
                    auto vecB = builder->create<mlir::vector::LoadOp>(
                        loc, vecType, rhs,
                        mlir::ValueRange{createConstantIndex(k), createConstantIndex(j)}).getResult();
                    
                    // 使用 arith 方言的浮点运算
                    auto mulResult = builder->create<mlir::arith::MulFOp>(
                        loc, vecType, vecA, vecB).getResult();
                    accum = builder->create<mlir::arith::AddFOp>(
                        loc, vecType, accum, mulResult).getResult();
                }
                
                // 存储结果
                builder->create<mlir::vector::StoreOp>(
                    loc, accum, result,
                    mlir::ValueRange{createConstantIndex(i), createConstantIndex(j)});
            }
        }
        
        // 添加优化属性
        addVectorizationAttributes(result.getDefiningOp());
        addMemoryAccessAttributes(result.getDefiningOp());
    }
    
    return result;
}

void MLIRGen::addVectorizationAttributes(mlir::Operation* op) {
    if (!op) return;
    
    // 添加向量化提示属性
    op->setAttr("vector.enable", 
                builder->getBoolAttr(true));
}

void MLIRGen::addParallelizationAttributes(mlir::Operation* op) {
    if (!op) return;
    
    // 添加并行化提示属性
    op->setAttr("parallel.enable", 
                builder->getBoolAttr(true));
}

void MLIRGen::optimizeMemoryAccess(mlir::Operation* op) {
    if (!op) return;
    
    // 添加内存访问优化属性
    op->setAttr("memory.layout", 
                builder->getStringAttr("blocked"));
}

void MLIRGen::addTilingAttributes(mlir::Operation* op, int64_t tileSize) {
    if (!op) return;
    
    // 添加分块大小属性
    op->setAttr("tile.size", 
                builder->getI64IntegerAttr(tileSize));
}

} // namespace matrix