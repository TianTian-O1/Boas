// MLIRGenNodes.cpp - AST节点处理实现
#include "mlirops/MLIRGen.h"
#include "frontend/ASTImpl.h"
#include "frontend/BasicAST.h"
#include "frontend/ModuleAST.h"
#include "frontend/FunctionAST.h"
#include <iostream>
#include <memory>

using namespace matrix;

namespace matrix {

mlir::Value MLIRGen::generateMLIRForFunction(const FunctionAST* func) {
    if (!func) {
        std::cerr << "Error: Null function AST\n";
        return nullptr;
    }

    std::cerr << "[DEBUG] Generating MLIR for function: " << func->getName() << "\n";

    try {
        const auto& body = func->getBody();
        std::cerr << "[DEBUG] Processing function body with " << body.size() << " statements\n";

        mlir::Value lastValue;
        for (const auto& stmt : body) {
            if (!stmt) {
                std::cerr << "Warning: Null statement in function body\n";
                continue;
            }

            std::cerr << "[DEBUG] Processing statement kind: " << stmt->getKind() << "\n";
            lastValue = generateMLIRForNode(stmt.get());
            
            if (!lastValue) {
                std::cerr << "Warning: Failed to generate MLIR for statement\n";
            }
        }

        return lastValue;
    } catch (const std::exception& e) {
        std::cerr << "Error in generateMLIRForFunction: " << e.what() << "\n";
        return nullptr;
    }
}

mlir::Value MLIRGen::generateMLIRForImport(const ImportAST* import) {
    if (!import) {
        std::cerr << "Error: Null import AST\n";
        return nullptr;
    }

    std::cerr << "[DEBUG] Processing import: " << import->getModuleName() << "\n";
    
    return builder->create<mlir::arith::ConstantIntOp>(
        builder->getUnknownLoc(),
        0,
        32
    );
}

mlir::Value MLIRGen::generateMLIRForVariable(const VariableExprAST* expr) {
    auto it = symbolTable.find(expr->getName());
    if (it != symbolTable.end()) {
        return it->second;
    }
    
    llvm::errs() << "[DEBUG] Variable " << expr->getName() << " not found in symbol table\n";
    return nullptr;
}

mlir::Value MLIRGen::generateMLIRForAssignment(const AssignmentExprAST* expr) {
    auto value = generateMLIRForNode(expr->getValue());
    if (!value) return nullptr;
    
    symbolTable[expr->getName()] = value;
    return value;
}

mlir::Value MLIRGen::generateNumberMLIR(const NumberExprAST* number) {
    return createConstantF64(number->getValue());
}

mlir::Value MLIRGen::generateMLIRForBinary(const BinaryExprAST* expr) {
    llvm::errs() << "[DEBUG] Processing Binary node\n";
    llvm::errs() << "[DEBUG] Binary operator: '" << expr->getOp() << "'\n";
    
    if (expr->getOp() == "matmul") {
        llvm::errs() << "[DEBUG] Found matmul operation\n";
        
        auto* lhs = static_cast<const VariableExprAST*>(expr->getLHS());
        auto* rhs = static_cast<const VariableExprAST*>(expr->getRHS());
        
        llvm::errs() << "[DEBUG] Matmul operands: " << lhs->getName() << " * " << rhs->getName() << "\n";
        
        auto lhsValue = symbolTable[lhs->getName()];
        auto rhsValue = symbolTable[rhs->getName()];
        
        if (!lhsValue || !rhsValue) {
            llvm::errs() << "[DEBUG] Failed to find operands in symbol table\n";
            return nullptr;
        }
        
        auto matmulExpr = std::make_unique<MatmulExprASTImpl>(
            std::make_unique<VariableExprASTImpl>("temp_lhs"),
            std::make_unique<VariableExprASTImpl>("temp_rhs")
        );
        
        auto result = generateMLIRForMatmul(matmulExpr.get());
        
        if (!result || !mlir::isa<mlir::MemRefType>(result.getType())) {
            llvm::errs() << "[DEBUG] Matmul result must be memref type\n";
            return nullptr;
        }
        
        return result;
    }
    
    return nullptr;
}

mlir::Value MLIRGen::generateMLIRForCall(const CallExprAST* expr) {
    std::cerr << "[DEBUG] Generating Call operation\n";
    const std::string& callee = expr->getCallee();
    
    if (callee == "matmul") {
        if (expr->getArgs().size() != 2) {
            std::cerr << "Error: matmul requires exactly 2 arguments\n";
            return nullptr;
        }
        auto lhs = generateMLIRForNode(expr->getArgs()[0].get());
        auto rhs = generateMLIRForNode(expr->getArgs()[1].get());
        if (!lhs || !rhs) return nullptr;
        
        auto matmulExpr = std::make_unique<MatmulExprASTImpl>(
            std::make_unique<VariableExprASTImpl>("temp_lhs"),
            std::make_unique<VariableExprASTImpl>("temp_rhs")
        );
        return generateMLIRForMatmul(matmulExpr.get());
    }
    
    std::cerr << "Error: Unknown function call: " << callee << "\n";
    return nullptr;
}

mlir::Value MLIRGen::generateMLIRForArray(const ArrayExprAST* expr) {
    std::cerr << "[DEBUG] Generating Array operation\n";
    // Implementation for array operations - to be implemented
    return nullptr;
}

mlir::Value MLIRGen::generateMLIRForPrint(const PrintExprAST* expr) {
    if (!expr) {
        std::cerr << "Null print expression\n";
        return nullptr;
    }
    
    auto value = generateMLIRForNode(expr->getValue());
    if (!value) {
        std::cerr << "Failed to generate value for print\n";
        return nullptr;
    }
    
    auto loc = builder->getUnknownLoc();
    
    // Handle different types
    if (auto numberExpr = llvm::dyn_cast<mlir::arith::ConstantFloatOp>(
            value.getDefiningOp())) {
        // For constant float values (like benchmark markers)
        builder->create<mlir::func::CallOp>(
            loc,
            "printFloat",
            mlir::TypeRange{},
            mlir::ValueRange{value}
        );
    } else if (auto intConst = llvm::dyn_cast<mlir::arith::ConstantIntOp>(
            value.getDefiningOp())) {
        // For constant integer values
        auto floatValue = builder->create<mlir::arith::SIToFPOp>(
            loc,
            builder->getF64Type(),
            value
        );
        
        builder->create<mlir::func::CallOp>(
            loc,
            "printFloat",
            mlir::TypeRange{},
            mlir::ValueRange{floatValue}
        );
    } else if (auto memrefTy = mlir::dyn_cast<mlir::MemRefType>(value.getType())) {
        // 将结果矩阵转换为 unranked memref 并打印
        auto resultType = mlir::UnrankedMemRefType::get(
            builder->getF64Type(),
            0
        );
        auto cast = builder->create<mlir::memref::CastOp>(
            loc,
            resultType,
            value
        );
        
        builder->create<mlir::func::CallOp>(
            loc,
            "printMemrefF64",
            mlir::TypeRange{},
            mlir::ValueRange{cast}
        );
    } else if (auto floatTy = mlir::dyn_cast<mlir::FloatType>(value.getType())) {
        // For other float values
        builder->create<mlir::func::CallOp>(
            loc,
            "printFloat", 
            mlir::TypeRange{},
            mlir::ValueRange{value}
        );
    } else {
        std::cerr << "Unsupported type for print\n";
        return nullptr;
    }
    
    return value;
}


} // namespace matrix