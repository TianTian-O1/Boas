// MLIRGenUtils.cpp - 工具函数实现
#include "mlirops/MLIRGen.h"

namespace matrix {

mlir::FloatType MLIRGen::getF64Type() {
    return builder->getF64Type();
}

mlir::Value MLIRGen::createConstantF64(double value) {
    return builder->create<mlir::arith::ConstantFloatOp>(
        builder->getUnknownLoc(),
        llvm::APFloat(value),
        getF64Type()
    );
}

mlir::Value MLIRGen::createConstantIndex(int64_t value) {
    return builder->create<mlir::arith::ConstantIndexOp>(
        builder->getUnknownLoc(),
        value
    );
}

mlir::MemRefType MLIRGen::getMemRefType(int64_t rows, int64_t cols) {
    return mlir::MemRefType::get({rows, cols}, getF64Type());
}

bool MLIRGen::isStoredInSymbolTable(mlir::Value value) {
    for (const auto& entry : symbolTable) {
        if (entry.second == value) {
            return true;
        }
    }
    return false;
}

void MLIRGen::dumpState(const std::string& message) {
    llvm::errs() << "\n=== " << message << " ===\n";
    if (module) {
        llvm::errs() << "Current MLIR Module:\n";
        module.dump();
    } else {
        llvm::errs() << "No module available\n";
    }
    llvm::errs() << "\nSymbol table contents:\n";
    for (const auto& entry : symbolTable) {
        llvm::errs() << "  " << entry.first << ": ";
        if (entry.second) {
            entry.second.getType().dump();
        } else {
            llvm::errs() << "null value";
        }
        llvm::errs() << "\n";
    }
    llvm::errs() << "==================\n\n";
}

} // namespace matrix