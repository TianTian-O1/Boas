#ifndef MATRIX_MLIR_GEN_H
#define MATRIX_MLIR_GEN_H

#include "frontend/AST.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include <memory>
#include <map>

namespace matrix {

class MLIRGen {
public:
    MLIRGen();
    mlir::ModuleOp generateMLIR(const std::vector<std::unique_ptr<ExprAST>>& ast);
    std::string getMLIRString(mlir::ModuleOp module);
    
    // AST 节点处理方法
    mlir::Value generateMLIRForNode(const ExprAST* node);
    mlir::Value generateMLIRForVariable(const VariableExprAST* expr);
    mlir::Value generateMLIRForAssignment(const AssignmentExprAST* expr);
    mlir::Value generateNumberMLIR(const NumberExprAST* number);
    mlir::Value generateMLIRForMatmul(const MatmulExprAST* expr);
    mlir::Value generateMLIRForPrint(const PrintExprAST* print);
    mlir::Value generateMLIRForTensorCreate(const TensorCreateExprAST* expr);

    // 辅助方法
    mlir::FloatType getF64Type();
    mlir::Value createConstantF64(double value);
    mlir::Value createConstantIndex(int64_t value);
    mlir::MemRefType getMemRefType(int64_t rows, int64_t cols);

private:
    std::unique_ptr<mlir::MLIRContext> context;
    std::unique_ptr<mlir::OpBuilder> builder;
    mlir::ModuleOp module;
    std::map<std::string, mlir::Value> symbolTable;

    mlir::Value generateMLIRForFunction(const FunctionAST* expr);
    mlir::Value generateMLIRForImport(const ImportAST* expr);
    mlir::Value generateMLIRForBinary(const BinaryExprAST* expr);

};

} // namespace matrix

#endif // MATRIX_MLIR_GEN_H
