#include "mlirops/NPUBackend.h"
#include "mlirops/CANNRuntime.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include <iostream>

namespace matrix {

// NPU-optimized matrix multiplication implementation using BiShengIR
class NPUMatmulOptimizer {
public:
    static mlir::Value generateOptimizedMatmul(
        mlir::OpBuilder& builder,
        mlir::Location loc,
        mlir::Value lhs,
        mlir::Value rhs,
        mlir::Value M,
        mlir::Value N,
        mlir::Value K) {
        
        std::cout << "[NPU Optimizer] Generating optimized matmul for Ascend NPU" << std::endl;
        
        // Get tensor types
        auto lhsType = lhs.getType().cast<mlir::MemRefType>();
        auto rhsType = rhs.getType().cast<mlir::MemRefType>();
        auto elementType = lhsType.getElementType();
        
        // Create result buffer with NPU-aligned memory
        auto resultType = mlir::MemRefType::get(
            {mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic},
            elementType
        );
        
        auto result = builder.create<mlir::memref::AllocOp>(
            loc, resultType, mlir::ValueRange{M, N}
        );
        
        // Add NPU-specific attributes for optimization
        auto npuAttr = builder.getStringAttr("npu_optimized");
        auto tileAttr = builder.getI64ArrayAttr({16, 16, 16}); // NPU tile sizes
        
        // Create optimized matmul operation
        auto matmulOp = builder.create<mlir::linalg::MatmulOp>(
            loc,
            mlir::ValueRange{lhs, rhs},
            mlir::ValueRange{result}
        );
        
        // Set NPU optimization attributes
        matmulOp->setAttr("device", npuAttr);
        matmulOp->setAttr("tile_sizes", tileAttr);
        matmulOp->setAttr("use_cube_unit", builder.getBoolAttr(true));
        
        // For large matrices, use Cube unit optimization
        auto mInt = builder.create<mlir::arith::IndexCastOp>(
            loc, builder.getI64Type(), M
        );
        auto threshold = builder.create<mlir::arith::ConstantIntOp>(
            loc, 256, builder.getI64Type()
        );
        auto cmp = builder.create<mlir::arith::CmpIOp>(
            loc, mlir::arith::CmpIPredicate::sge, mInt, threshold
        );
        
        // Add conditional optimization for large matrices
        builder.create<mlir::scf::IfOp>(
            loc,
            cmp,
            [&](mlir::OpBuilder& b, mlir::Location l) {
                // Large matrix path - use Cube unit
                matmulOp->setAttr("algorithm", b.getStringAttr("cube_gemm"));
                b.create<mlir::scf::YieldOp>(l);
            },
            [&](mlir::OpBuilder& b, mlir::Location l) {
                // Small matrix path - use Vector unit
                matmulOp->setAttr("algorithm", b.getStringAttr("vector_gemm"));
                b.create<mlir::scf::YieldOp>(l);
            }
        );
        
        return result;
    }
    
    // Optimize existing matmul operations for NPU
    static void optimizeMatmulPass(mlir::func::FuncOp func) {
        std::cout << "[NPU Optimizer] Running matmul optimization pass" << std::endl;
        
        func.walk([&](mlir::linalg::MatmulOp op) {
            // Check if already optimized
            if (op->hasAttr("npu_optimized")) {
                return;
            }
            
            mlir::OpBuilder builder(op);
            auto loc = op.getLoc();
            
            // Get operands
            auto lhs = op.getInputs()[0];
            auto rhs = op.getInputs()[1];
            auto result = op.getOutputs()[0];
            
            // Add NPU optimization attributes
            op->setAttr("npu_optimized", builder.getBoolAttr(true));
            op->setAttr("device", builder.getStringAttr("ascend"));
            
            // Determine optimal tiling based on matrix size
            auto lhsType = lhs.getType().cast<mlir::MemRefType>();
            auto shape = lhsType.getShape();
            
            // Use different tile sizes based on matrix dimensions
            if (shape.size() >= 2 && shape[0] != mlir::ShapedType::kDynamic) {
                if (shape[0] >= 512) {
                    // Large matrices - use 64x64 tiles
                    op->setAttr("tile_sizes", builder.getI64ArrayAttr({64, 64, 64}));
                    op->setAttr("algorithm", builder.getStringAttr("cube_gemm"));
                } else if (shape[0] >= 128) {
                    // Medium matrices - use 32x32 tiles
                    op->setAttr("tile_sizes", builder.getI64ArrayAttr({32, 32, 32}));
                    op->setAttr("algorithm", builder.getStringAttr("mixed_gemm"));
                } else {
                    // Small matrices - use 16x16 tiles
                    op->setAttr("tile_sizes", builder.getI64ArrayAttr({16, 16, 16}));
                    op->setAttr("algorithm", builder.getStringAttr("vector_gemm"));
                }
            }
            
            // Add data layout optimization hints
            op->setAttr("layout", builder.getStringAttr("ND"));
            op->setAttr("use_fp16", builder.getBoolAttr(false)); // Use FP64 for accuracy
            
            std::cout << "[NPU Optimizer] Optimized matmul operation" << std::endl;
        });
    }
    
    // Generate fusion opportunities for multiple matmuls
    static void fuseMa tmuls(mlir::func::FuncOp func) {
        std::cout << "[NPU Optimizer] Looking for matmul fusion opportunities" << std::endl;
        
        std::vector<mlir::linalg::MatmulOp> matmuls;
        func.walk([&](mlir::linalg::MatmulOp op) {
            matmuls.push_back(op);
        });
        
        // Check for fusable patterns (A*B)*C -> fused_gemm(A,B,C)
        for (size_t i = 0; i < matmuls.size(); ++i) {
            for (size_t j = i + 1; j < matmuls.size(); ++j) {
                auto op1 = matmuls[i];
                auto op2 = matmuls[j];
                
                // Check if output of op1 is input to op2
                if (op1.getResult(0) == op2.getInputs()[0]) {
                    std::cout << "[NPU Optimizer] Found fusable matmul chain" << std::endl;
                    
                    mlir::OpBuilder builder(op2);
                    // Mark for fusion
                    op1->setAttr("fuse_next", builder.getBoolAttr(true));
                    op2->setAttr("fused_with_prev", builder.getBoolAttr(true));
                }
            }
        }
    }
};

// Pass to optimize matmul operations for NPU
struct NPUMatmulOptimizationPass 
    : public mlir::PassWrapper<NPUMatmulOptimizationPass, mlir::OperationPass<mlir::func::FuncOp>> {
    
    void runOnOperation() override {
        auto func = getOperation();
        
        // Initialize NPU backend if available
        if (!NPUBackend::isAvailable()) {
            std::cout << "[NPU Pass] NPU not available, skipping optimization" << std::endl;
            return;
        }
        
        // Run optimization passes
        NPUMatmulOptimizer::optimizeMatmulPass(func);
        NPUMatmulOptimizer::fuseMatmuls(func);
        
        std::cout << "[NPU Pass] Optimization complete" << std::endl;
    }
};

// Register the pass
static mlir::PassRegistration<NPUMatmulOptimizationPass>
    registration("npu-matmul-opt", "Optimize matmul operations for Ascend NPU");

} // namespace matrix