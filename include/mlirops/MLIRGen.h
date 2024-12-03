// MLIRGen.h - 主头文件
#ifndef MATRIX_MLIR_GEN_H
#define MATRIX_MLIR_GEN_H

#include <algorithm>
#include "frontend/AST.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/ADT/APFloat.h"

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
    std::vector<std::map<std::string, mlir::Value>> scopeStack;

    // Core methods
    void initializeContext();
    void declareRuntimeFunctions();
    mlir::Value generateMLIRForNode(const ExprAST* node);
    void processASTNode(mlir::Block* block, const std::vector<std::unique_ptr<ExprAST>>& ast);

    // AST node handlers - defined in MLIRGenNodes.cpp
    mlir::Value generateMLIRForFunction(const FunctionAST* func);
    void handleFunctionReturn(const FunctionAST* func, mlir::Value result, mlir::Block* entryBlock);
    void handleDefaultReturn(const FunctionAST* func, mlir::Block* entryBlock);
    void initializeMemRefToZero(mlir::Value memref, mlir::Value rows, mlir::Value cols);
    mlir::Value generateMLIRForImport(const ImportAST* import);
    mlir::Value generateMLIRForVariable(const VariableExprAST* expr); 
    mlir::Value generateMLIRForAssignment(const AssignmentExprAST* expr);
    mlir::Value generateNumberMLIR(const NumberExprAST* number);
    mlir::Value generateMLIRForBinary(const BinaryExprAST* expr);
    mlir::Value generateMLIRForCall(const CallExprAST* expr);
    mlir::Value generateMLIRForArray(const ArrayExprAST* expr);
    mlir::Value generateMLIRForPrint(const PrintExprAST* print);

    // Tensor operations - defined in MLIRGenTensor.cpp
    mlir::Value generateMLIRForTensor(const TensorExprAST* expr);
    mlir::Value generateMLIRForTensorCreate(const TensorCreateExprAST* expr);
    mlir::Value generateMLIRForTensorRandom(const TensorRandomExprAST* expr);

    // Matrix operations - defined in MLIRGenMatrix.cpp 
    mlir::Value generateMLIRForMatmul(const MatmulExprAST* expr);
    mlir::Value createOptimizedMatmul(mlir::Value lhs, mlir::Value rhs, mlir::Value result,
                                     mlir::Value M, mlir::Value N, mlir::Value K);
    mlir::Value createBlockedMatmul(mlir::Value lhs, mlir::Value rhs, const MatmulExprAST* expr);
    mlir::Value createVectorizedMatmul(mlir::Value lhs, mlir::Value rhs, const MatmulExprAST* expr);

    // Timing operations - defined in MLIRGenTiming.cpp
    mlir::Value generateTimeNowMLIR(const TimeCallExprAST* expr);
    mlir::Value generateTimeDiffMLIR(mlir::Value lhs, mlir::Value rhs);
    mlir::Value convertToMilliseconds(mlir::Value seconds);

    // Helper methods - defined in MLIRGenUtils.cpp
    mlir::FloatType getF64Type();
    mlir::Value createConstantF64(double value);
    mlir::Value createConstantIndex(int64_t value); 
    mlir::MemRefType getMemRefType(int64_t rows, int64_t cols);
    bool isStoredInSymbolTable(mlir::Value value);
    void dumpState(const std::string& message);
    
    // Optimization helpers - defined in MLIRGenOptimizations.cpp
    void addVectorizationAttributes(mlir::Operation* op);
    void addParallelizationAttributes(mlir::Operation* op);
    void optimizeMemoryAccess(mlir::Operation* op);
    void addTilingAttributes(mlir::Operation* op, int64_t tileSize);

    // Constants
    static constexpr int64_t TILE_SIZE = 32;
    static constexpr int64_t VECTOR_SIZE = 8;
    static constexpr int64_t CACHE_LINE_SIZE = 64;

    // 小矩阵优化相关函数
    mlir::Value optimizeSmallMatrixMultiply(mlir::Value lhs, mlir::Value rhs, 
                                           mlir::Value result, 
                                           int64_t M, int64_t N, int64_t K);

    // 添加新的私有方法
    mlir::Value createFullyUnrolledMatmul(mlir::Value lhs, mlir::Value rhs, 
                                         mlir::Value result,
                                         int64_t M, int64_t N, int64_t K);
    void addMemoryAccessAttributes(mlir::Operation* op);

    // 添加列表相关的辅助方法
    mlir::Value createList(const std::vector<mlir::Value>& elements);
    mlir::Value getListElement(mlir::Value list, mlir::Value index);
    mlir::Value setListElement(mlir::Value list, mlir::Value index, mlir::Value value);
    
    // 添加 generate 函数声明
    mlir::Value generate(const ExprAST* expr);

    mlir::Value generateListIndex(const ListIndexExprAST* expr);

public:
    // 列表操作的公共接口
    mlir::Value generateList(const ListExprAST* list);
    void pushScope();
    void popScope();
};

} // namespace matrix

#endif // MATRIX_MLIR_GEN_H