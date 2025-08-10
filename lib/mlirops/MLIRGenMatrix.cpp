// MLIRGenMatrix.cpp - Matrix operations MLIR generation with NPU optimization
#include "mlirops/MLIRGen.h"
#include "mlirops/NPUBackend.h"
#include "frontend/AST.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include <iostream>

namespace matrix {

/// Generate MLIR for matrix operations
mlir::Value MLIRGen::generateMLIRForMatmul(const MatmulExprAST* expr) {
    std::cerr << "[DEBUG] Generating MLIR for matrix operation\n";
    
    // Generate MLIR for LHS and RHS operands
    mlir::Value lhs = generateMLIRForNode(expr->getLHS());
    mlir::Value rhs = generateMLIRForNode(expr->getRHS());
    
    if (!lhs || !rhs) {
        std::cerr << "Failed to generate operands for matrix operation\n";
        return nullptr;
    }
    
    // Get operand types
    auto lhsType = llvm::dyn_cast<mlir::MemRefType>(lhs.getType());
    auto rhsType = llvm::dyn_cast<mlir::MemRefType>(rhs.getType());
    
    if (!lhsType || !rhsType) {
        std::cerr << "Matrix operands must be memref types\n";
        return nullptr;
    }
    
    // Extract dimensions for matrix multiplication (A[m,k] * B[k,n] = C[m,n])
    auto lhsShape = lhsType.getShape();
    auto rhsShape = rhsType.getShape();
    
    if (lhsShape.size() != 2 || rhsShape.size() != 2) {
        std::cerr << "Matrix operands must be 2D\n";
        return nullptr;
    }
    
    // Create dimension values
    auto loc = builder->getUnknownLoc();
    auto indexType = builder->getIndexType();
    
    mlir::Value m, n, k1, k2;
    
    if (lhsShape[0] == mlir::ShapedType::kDynamic) {
        m = builder->create<mlir::memref::DimOp>(loc, lhs, 0);
    } else {
        m = builder->create<mlir::arith::ConstantIndexOp>(loc, lhsShape[0]);
    }
    
    if (lhsShape[1] == mlir::ShapedType::kDynamic) {
        k1 = builder->create<mlir::memref::DimOp>(loc, lhs, 1);
    } else {
        k1 = builder->create<mlir::arith::ConstantIndexOp>(loc, lhsShape[1]);
    }
    
    if (rhsShape[0] == mlir::ShapedType::kDynamic) {
        k2 = builder->create<mlir::memref::DimOp>(loc, rhs, 0);
    } else {
        k2 = builder->create<mlir::arith::ConstantIndexOp>(loc, rhsShape[0]);
    }
    
    if (rhsShape[1] == mlir::ShapedType::kDynamic) {
        n = builder->create<mlir::memref::DimOp>(loc, rhs, 1);
    } else {
        n = builder->create<mlir::arith::ConstantIndexOp>(loc, rhsShape[1]);
    }
    
    // Create result memref
    auto resultType = mlir::MemRefType::get({lhsShape[0], rhsShape[1]}, lhsType.getElementType());
    auto result = builder->create<mlir::memref::AllocOp>(loc, resultType);
    
    // Choose optimization strategy - check for NPU availability
    mlir::Value optimizedResult;
    if (NPUBackend::isAvailable()) {
        std::cerr << "[DEBUG] Using NPU-optimized matrix multiplication\n";
        optimizedResult = NPUBackend::generateNPUMatmul(this, lhs, rhs, m, n, k1);
    } else {
        std::cerr << "[DEBUG] Using CPU-optimized matrix multiplication\n";
        optimizedResult = createOptimizedMatmul(lhs, rhs, result, m, n, k1);
    }
    
    if (!optimizedResult) {
        std::cerr << "Failed to generate optimized matrix multiplication\n";
        return nullptr;
    }
    
    std::cerr << "[DEBUG] Matrix operation MLIR generation completed\n";
    return optimizedResult;
}

/// Create CPU-optimized matrix multiplication with loop tiling
mlir::Value MLIRGen::createOptimizedMatmul(mlir::Value lhs, mlir::Value rhs, 
                                          mlir::Value result, mlir::Value M, 
                                          mlir::Value N, mlir::Value K) {
    auto loc = builder->getUnknownLoc();
    auto indexType = builder->getIndexType();
    
    // Create constants for loop bounds
    auto zero = builder->create<mlir::arith::ConstantIndexOp>(loc, 0);
    auto one = builder->create<mlir::arith::ConstantIndexOp>(loc, 1);
    
    // Tiling factors for better cache performance
    auto tileSize = builder->create<mlir::arith::ConstantIndexOp>(loc, 32);
    
    // Initialize result matrix with zeros
    auto elementType = mlir::cast<mlir::MemRefType>(result.getType()).getElementType();
    auto zero_f = builder->create<mlir::arith::ConstantOp>(loc, elementType, 
                     builder->getZeroAttr(elementType));
    
    // Create tiled matrix multiplication loops
    auto outerLoopI = builder->create<mlir::scf::ForOp>(loc, zero, M, tileSize);
    {
        mlir::OpBuilder::InsertionGuard guard(*builder);
        builder->setInsertionPointToStart(outerLoopI.getBody());
        auto i_tile = outerLoopI.getInductionVar();
        
        auto outerLoopJ = builder->create<mlir::scf::ForOp>(loc, zero, N, tileSize);
        {
            mlir::OpBuilder::InsertionGuard guard(*builder);
            builder->setInsertionPointToStart(outerLoopJ.getBody());
            auto j_tile = outerLoopJ.getInductionVar();
            
            auto outerLoopK = builder->create<mlir::scf::ForOp>(loc, zero, K, tileSize);
            {
                mlir::OpBuilder::InsertionGuard guard(*builder);
                builder->setInsertionPointToStart(outerLoopK.getBody());
                auto k_tile = outerLoopK.getInductionVar();
                
                // Calculate tile bounds
                auto i_end = builder->create<mlir::arith::MinSIOp>(loc, 
                    builder->create<mlir::arith::AddIOp>(loc, i_tile, tileSize), M);
                auto j_end = builder->create<mlir::arith::MinSIOp>(loc, 
                    builder->create<mlir::arith::AddIOp>(loc, j_tile, tileSize), N);
                auto k_end = builder->create<mlir::arith::MinSIOp>(loc, 
                    builder->create<mlir::arith::AddIOp>(loc, k_tile, tileSize), K);
                
                // Inner loops for the tile
                auto innerLoopI = builder->create<mlir::scf::ForOp>(loc, i_tile, i_end, one);
                {
                    mlir::OpBuilder::InsertionGuard guard(*builder);
                    builder->setInsertionPointToStart(innerLoopI.getBody());
                    auto i = innerLoopI.getInductionVar();
                    
                    auto innerLoopJ = builder->create<mlir::scf::ForOp>(loc, j_tile, j_end, one);
                    {
                        mlir::OpBuilder::InsertionGuard guard(*builder);
                        builder->setInsertionPointToStart(innerLoopJ.getBody());
                        auto j = innerLoopJ.getInductionVar();
                        
                        // Load initial accumulator
                        auto acc_init = builder->create<mlir::memref::LoadOp>(loc, result, 
                                                                            mlir::ValueRange{i, j});
                        
                        auto innerLoopK = builder->create<mlir::scf::ForOp>(loc, k_tile, k_end, one, 
                                                                          mlir::ValueRange{acc_init});
                        {
                            mlir::OpBuilder::InsertionGuard guard(*builder);
                            builder->setInsertionPointToStart(innerLoopK.getBody());
                            auto k = innerLoopK.getInductionVar();
                            auto acc = innerLoopK.getRegionIterArgs()[0];
                            
                            // Load operands
                            auto a_val = builder->create<mlir::memref::LoadOp>(loc, lhs, 
                                                                             mlir::ValueRange{i, k});
                            auto b_val = builder->create<mlir::memref::LoadOp>(loc, rhs, 
                                                                             mlir::ValueRange{k, j});
                            
                            // Compute: acc += a * b
                            auto prod = builder->create<mlir::arith::MulFOp>(loc, a_val, b_val);
                            auto new_acc = builder->create<mlir::arith::AddFOp>(loc, acc, prod);
                            
                            builder->create<mlir::scf::YieldOp>(loc, mlir::ValueRange{new_acc});
                        }
                        
                        // Store final result
                        auto final_acc = innerLoopK.getResult(0);
                        builder->create<mlir::memref::StoreOp>(loc, final_acc, result, 
                                                             mlir::ValueRange{i, j});
                    }
                }
            }
        }
    }
    
    return result;
}

} // namespace matrix
