#ifndef MATRIX_MLIR_GEN_H
#define MATRIX_MLIR_GEN_H

#include <algorithm>
#include "frontend/AST.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"        // SCF dialect
#include "mlir/Dialect/Vector/IR/VectorOps.h" // Vector dialect
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace matrix {

class MLIRGen {
public:
    MLIRGen();
    mlir::ModuleOp generateMLIR(const std::vector<std::unique_ptr<ExprAST>>& ast);
    std::string getMLIRString(mlir::ModuleOp module);

private:
    std::unique_ptr<mlir::MLIRContext> context;
    std::unique_ptr<mlir::OpBuilder> builder;
    mlir::ModuleOp module;
    std::map<std::string, mlir::Value> symbolTable;

    // 优化相关的常量
    static constexpr int64_t TILE_SIZE = 32;
    static constexpr int64_t VECTOR_SIZE = 8;
    static constexpr int64_t CACHE_LINE_SIZE = 64;

    // Debugging helper
    void dumpState(const std::string& message);

    // AST node handlers
    mlir::Value generateMLIRForNode(const ExprAST* node);
    mlir::Value generateMLIRForFunction(const FunctionAST* func);
    mlir::Value generateMLIRForImport(const ImportAST* import);
    mlir::Value generateMLIRForVariable(const VariableExprAST* expr);
    mlir::Value generateMLIRForAssignment(const AssignmentExprAST* expr);
    mlir::Value generateNumberMLIR(const NumberExprAST* number);
    mlir::Value generateMLIRForBinary(const BinaryExprAST* expr);
    mlir::Value generateMLIRForCall(const CallExprAST* expr);
    mlir::Value generateMLIRForArray(const ArrayExprAST* expr);
    mlir::Value generateMLIRForTensor(const TensorExprAST* expr);
    mlir::Value generateMLIRForTensorCreate(const TensorCreateExprAST* expr);
    mlir::Value generateMLIRForMatmul(const MatmulExprAST* expr);
    mlir::Value generateMLIRForTensorRandom(const TensorRandomExprAST* expr);
    mlir::Value generateMLIRForPrint(const PrintExprAST* print);

    // Helper methods
    mlir::FloatType getF64Type();
    mlir::Value createConstantF64(double value);
    mlir::Value createConstantIndex(int64_t value);
    mlir::MemRefType getMemRefType(int64_t rows, int64_t cols);
    void processASTNode(mlir::Block* block, const std::vector<std::unique_ptr<ExprAST>>& ast);
    bool isStoredInSymbolTable(mlir::Value value);
    
    // Optimization helpers
    mlir::Value createOptimizedMatmul(mlir::Value lhs, mlir::Value rhs, mlir::Value result,
                                     int64_t M, int64_t N, int64_t K);
    void addVectorizationAttributes(mlir::Operation* op);
    void addParallelizationAttributes(mlir::Operation* op);
    
    // Matrix operation optimization helpers
    mlir::Value createBlockedMatmul(mlir::Value lhs, mlir::Value rhs, const MatmulExprAST* expr);
    mlir::Value createVectorizedMatmul(mlir::Value lhs, mlir::Value rhs, const MatmulExprAST* expr);
    void optimizeMemoryAccess(mlir::Operation* op);
    void addTilingAttributes(mlir::Operation* op, int64_t tileSize);
};

} // namespace matrix

#endif // MATRIX_MLIR_GEN_H